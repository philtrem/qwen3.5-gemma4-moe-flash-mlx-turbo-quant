pub mod attention;
pub mod gemma4_attention;
pub mod gated_delta;
pub mod mlp;
pub mod moe;
pub mod norm;

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use mlx_rs::error::Exception;
use mlx_rs::Array;

use crate::cache::{ArraysCache, Cache, KVCache};
use crate::config::{ModelType, TextModelArgs};
use crate::memory::ExpertMemoryManager;
use crate::perf::PerfStats;
use attention::Attention;
use gemma4_attention::Gemma4Attention;
use gated_delta::GatedDeltaNet;
use std::cell::RefCell;
use mlp::{QuantizedLinear, GeLUMLP, MLP};
use moe::{Gemma4MoeBlock, SparseMoeBlock, TransitionProfiler};
use norm::RMSNorm;

// --- Qwen layer types ---

pub enum AttentionLayer {
    Linear(GatedDeltaNet),
    Full(Attention),
    Gemma4(Gemma4Attention),
}

pub struct DecoderLayer {
    pub attention: AttentionLayer,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
    pub mlp: MoeVariant,
    // Gemma4 extra norms
    pub pre_feedforward_layernorm: Option<RMSNorm>,
    pub post_feedforward_layernorm: Option<RMSNorm>,
    pub post_feedforward_layernorm_1: Option<RMSNorm>,
    pub post_feedforward_layernorm_2: Option<RMSNorm>,
    pub pre_feedforward_layernorm_2: Option<RMSNorm>,
    // Gemma4 dense MLP (runs in parallel with MoE)
    pub dense_mlp: Option<GeLUMLP>,
    // Gemma4 layer scalar
    pub layer_scalar: Option<Array>,
}

pub enum MoeVariant {
    Qwen(SparseMoeBlock),
    Gemma4(Gemma4MoeBlock),
}

impl DecoderLayer {
    pub fn is_linear(&self) -> bool {
        matches!(&self.attention, AttentionLayer::Linear(_))
    }

    pub fn is_gemma4(&self) -> bool {
        matches!(&self.attention, AttentionLayer::Gemma4(_))
    }

    pub fn kv_head_dim(&self) -> usize {
        match &self.attention {
            AttentionLayer::Full(a) => a.head_dim,
            AttentionLayer::Gemma4(a) => a.head_dim,
            AttentionLayer::Linear(_) => 0, // not used
        }
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut Cache,
        mem: &ExpertMemoryManager,
        perf: &PerfStats,
        tp: Option<&RefCell<TransitionProfiler>>,
    ) -> Result<Array, Exception> {
        if self.is_gemma4() {
            self.forward_gemma4(x, mask, cache, mem, perf, tp)
        } else {
            self.forward_qwen(x, mask, cache, mem, perf, tp)
        }
    }

    fn forward_qwen(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut Cache,
        mem: &ExpertMemoryManager,
        perf: &PerfStats,
        tp: Option<&RefCell<TransitionProfiler>>,
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = match &mut self.attention {
            AttentionLayer::Linear(gdn) => {
                gdn.forward(&normed, mask, cache.as_arrays_mut(), perf)?
            }
            AttentionLayer::Full(attn) => {
                attn.forward(&normed, mask, cache.as_kv_mut())?
            }
            _ => unreachable!(),
        };
        let h = x + &attn_out;
        let normed = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = match &self.mlp {
            MoeVariant::Qwen(moe) => moe.forward(&normed, mem, perf, tp)?,
            _ => unreachable!(),
        };
        Ok(&h + &mlp_out)
    }

    fn forward_gemma4(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut Cache,
        mem: &ExpertMemoryManager,
        perf: &PerfStats,
        tp: Option<&RefCell<TransitionProfiler>>,
    ) -> Result<Array, Exception> {
        // Attention block
        let residual = x.clone();
        let h = self.input_layernorm.forward(x)?;
        let h = match &mut self.attention {
            AttentionLayer::Gemma4(attn) => {
                attn.forward(&h, mask, cache.as_kv_mut())?
            }
            _ => unreachable!(),
        };
        let h = self.post_attention_layernorm.forward(&h)?;
        let h = &residual + &h;

        // Feedforward block: dense MLP + MoE in parallel
        let residual = h.clone();

        // Dense MLP path
        let h1 = self.pre_feedforward_layernorm.as_ref().unwrap().forward(&h)?;
        let h1 = self.dense_mlp.as_ref().unwrap().forward(&h1)?;
        let h1 = self.post_feedforward_layernorm_1.as_ref().unwrap().forward(&h1)?;

        // MoE expert path
        let h2_input = self.pre_feedforward_layernorm_2.as_ref().unwrap().forward(&h)?;
        let h2 = match &self.mlp {
            MoeVariant::Gemma4(moe) => moe.forward(&h2_input, mem, perf, tp)?,
            _ => unreachable!(),
        };
        let h2 = self.post_feedforward_layernorm_2.as_ref().unwrap().forward(&h2)?;

        // Sum dense + expert
        let h = &h1 + &h2;

        let h = self.post_feedforward_layernorm.as_ref().unwrap().forward(&h)?;
        let mut h = &residual + &h;

        // Layer scalar
        if let Some(ref scalar) = self.layer_scalar {
            h = &h * scalar;
        }

        Ok(h)
    }
}

pub struct TextModel {
    pub embed_tokens_weight: Array,
    pub embed_tokens_scales: Option<Array>,
    pub embed_tokens_biases: Option<Array>,
    pub embed_bits: i32,
    pub embed_group_size: i32,
    pub layers: Vec<DecoderLayer>,
    pub norm: RMSNorm,
    pub full_attention_interval: usize,
    pub model_type: ModelType,
    pub embed_scale: Option<f32>,
}

impl TextModel {
    pub fn forward(
        &mut self,
        input_ids: &Array,
        cache: &mut [Cache],
        mem: &ExpertMemoryManager,
        perf: &PerfStats,
        speculate: bool,
        tp: Option<&RefCell<TransitionProfiler>>,
    ) -> Result<Array, Exception> {
        let hidden = match self.model_type {
            ModelType::Qwen => {
                let flat_ids = input_ids.flatten(None, None)?;
                let w = mlx_rs::ops::indexing::take_axis(&self.embed_tokens_weight, &flat_ids, 0)?;
                let s = mlx_rs::ops::indexing::take_axis(self.embed_tokens_scales.as_ref().unwrap(), &flat_ids, 0)?;
                let b = mlx_rs::ops::indexing::take_axis(self.embed_tokens_biases.as_ref().unwrap(), &flat_ids, 0)?;
                let hidden = mlx_rs::ops::dequantize(
                    &w, &s, &b, Some(self.embed_group_size), Some(self.embed_bits),
                )?;
                let shape = input_ids.shape();
                hidden.reshape(&[shape[0], shape[1], -1])?
            }
            ModelType::Gemma4 => {
                // Non-quantized embedding + scale by sqrt(hidden_size)
                let flat_ids = input_ids.flatten(None, None)?;
                let h = mlx_rs::ops::indexing::take_axis(&self.embed_tokens_weight, &flat_ids, 0)?;
                let shape = input_ids.shape();
                let h = h.reshape(&[shape[0], shape[1], -1])?;
                if let Some(scale) = self.embed_scale {
                    let s = Array::from_f32(scale).as_dtype(h.dtype())?;
                    &h * &s
                } else {
                    h
                }
            }
        };

        // Create attention masks
        let (full_mask, sliding_mask) = match self.model_type {
            ModelType::Qwen => {
                let fa_idx = self.full_attention_interval - 1;
                let fa_offset = cache[fa_idx].kv_offset();
                let mask = create_attention_mask(&hidden, fa_offset)?;
                (mask, None)
            }
            ModelType::Gemma4 => {
                // Find first full and first sliding attention layer for mask offsets
                let mut full_offset = 0usize;
                let mut sliding_offset = 0usize;
                for (i, layer) in self.layers.iter().enumerate() {
                    if matches!(&layer.attention, AttentionLayer::Gemma4(a) if !a.use_k_eq_v) {
                        // Sliding attention layer (has v_proj)
                        sliding_offset = cache[i].kv_offset();
                        break;
                    }
                }
                for (i, layer) in self.layers.iter().enumerate() {
                    if matches!(&layer.attention, AttentionLayer::Gemma4(a) if a.use_k_eq_v) {
                        // Full attention layer (K==V)
                        full_offset = cache[i].kv_offset();
                        break;
                    }
                }
                let full_mask = create_attention_mask(&hidden, full_offset)?;
                let sliding_mask = create_sliding_mask(&hidden, sliding_offset, self.layers[0].layer_scalar.is_some())?;
                (full_mask, sliding_mask)
            }
        };

        let mut h = hidden;
        let num_layers = self.layers.len();

        for i in 0..num_layers {
            let layer = &mut self.layers[i];
            let mask = match self.model_type {
                ModelType::Qwen => {
                    if layer.is_linear() { None } else { full_mask.as_ref() }
                }
                ModelType::Gemma4 => {
                    if matches!(&layer.attention, AttentionLayer::Gemma4(a) if a.use_k_eq_v) {
                        full_mask.as_ref()
                    } else {
                        sliding_mask.as_ref().or(full_mask.as_ref())
                    }
                }
            };
            h = layer.forward(&h, mask, &mut cache[i], mem, perf, tp)?;

            let _t = Instant::now();
            mlx_rs::transforms::async_eval(std::iter::once(&h))?;

            // Speculative prefetch via GCD utility queue
            if speculate && i + 1 < num_layers {
                if let Some(tp_ref) = tp {
                    let tp_borrow = tp_ref.borrow();
                    if let Some((pred_layer, ref predicted)) = tp_borrow.pending_prediction {
                        if pred_layer == i + 1 {
                            mem.prefetch_gcd_speculative(pred_layer, predicted);
                        }
                    }
                }
            }

            let _tw = Instant::now();
            mlx_rs::transforms::eval(std::iter::once(&h))?;
            perf.acc(&perf.eval_wait, _tw.elapsed());

            perf.acc(&perf.layer_eval, _t.elapsed());
        }

        self.norm.forward(&h)
    }
}

pub struct Model {
    pub model: TextModel,
    pub lm_head: QuantizedLinear,
    pub tie_word_embeddings: bool,
    pub head_dim: usize,
    pub final_logit_softcapping: Option<f32>,
}

impl Model {
    pub fn forward(
        &mut self,
        input_ids: &Array,
        cache: &mut [Cache],
        mem: &ExpertMemoryManager,
        perf: &PerfStats,
        speculate: bool,
        tp: Option<&RefCell<TransitionProfiler>>,
    ) -> Result<Array, Exception> {
        let out = self.model.forward(input_ids, cache, mem, perf, speculate, tp)?;
        let logits = if self.tie_word_embeddings {
            match self.model.model_type {
                ModelType::Qwen => {
                    mlx_rs::ops::quantized_matmul(
                        &out,
                        &self.model.embed_tokens_weight,
                        self.model.embed_tokens_scales.as_ref().unwrap(),
                        self.model.embed_tokens_biases.as_ref().unwrap(),
                        Some(true),
                        Some(self.model.embed_group_size),
                        Some(self.model.embed_bits),
                    )?
                }
                ModelType::Gemma4 => {
                    // Non-quantized embedding: matmul(out, weight.T)
                    let w_t = mlx_rs::ops::transpose_axes(&self.model.embed_tokens_weight, &[1, 0])?.as_dtype(out.dtype())?;
                    mlx_rs::ops::matmul(&out, &w_t)?
                }
            }
        } else {
            self.lm_head.forward(&out)?
        };

        // Logit softcapping (Gemma4)
        if let Some(softcap) = self.final_logit_softcapping {
            let s = Array::from_f32(softcap);
            let scaled = &logits / &s;
            let capped = mlx_rs::ops::tanh(&scaled)?;
            Ok(&capped * &s)
        } else {
            Ok(logits)
        }
    }

    pub fn make_cache(&self, kv_quant_bits: Option<u8>) -> Vec<Cache> {
        self.model
            .layers
            .iter()
            .map(|layer| {
                if layer.is_linear() {
                    Cache::Arrays(ArraysCache::new(2))
                } else {
                    match kv_quant_bits {
                        Some(bits) => Cache::KV(KVCache::new_quantized(
                            layer.kv_head_dim(), bits,
                        )),
                        None => Cache::KV(KVCache::new()),
                    }
                }
            })
            .collect()
    }
}

// --- Weight loading ---

pub fn load_model(split_path: &Path, args: &TextModelArgs, quant: Option<&crate::config::QuantizationConfig>) -> anyhow::Result<Model> {
    match args.model_type() {
        ModelType::Qwen => load_qwen_model(split_path, args, quant),
        ModelType::Gemma4 => load_gemma4_model(split_path, args, quant),
    }
}

fn load_qwen_model(split_path: &Path, args: &TextModelArgs, quant: Option<&crate::config::QuantizationConfig>) -> anyhow::Result<Model> {
    eprintln!("Loading resident weights (Qwen)...");
    let resident_path = split_path.join("resident/resident.safetensors");
    let weights = load_safetensors_map(&resident_path)?;

    eprintln!(
        "Loaded {} resident tensors ({:.2} GB)",
        weights.len(),
        weights.values().map(|a| a.nbytes()).sum::<usize>() as f64 / 1e9
    );

    let bits = quant.map(|q| q.bits as i32).unwrap_or(8);
    let group_size = quant.map(|q| q.group_size as i32).unwrap_or(32);
    eprintln!("  Quantization: {}-bit, group_size={}", bits, group_size);

    let mut layers = Vec::with_capacity(args.num_hidden_layers);
    for i in 0..args.num_hidden_layers {
        let prefix = format!("model.layers.{}", i);

        let input_ln = RMSNorm {
            weight: get_weight(&weights, &format!("{}.input_layernorm.weight", prefix)),
            eps: args.rms_norm_eps,
        };
        let post_ln = RMSNorm {
            weight: get_weight(&weights, &format!("{}.post_attention_layernorm.weight", prefix)),
            eps: args.rms_norm_eps,
        };

        let attention = if args.is_linear_layer(i) {
            let p = format!("{}.linear_attn", prefix);
            AttentionLayer::Linear(GatedDeltaNet {
                in_proj_qkv: load_qlinear(&weights, &format!("{}.in_proj_qkv", p), bits, group_size),
                in_proj_z: load_qlinear(&weights, &format!("{}.in_proj_z", p), bits, group_size),
                in_proj_b: load_qlinear(&weights, &format!("{}.in_proj_b", p), bits, group_size),
                in_proj_a: load_qlinear(&weights, &format!("{}.in_proj_a", p), bits, group_size),
                out_proj: load_qlinear(&weights, &format!("{}.out_proj", p), bits, group_size),
                conv1d_weight: get_weight(&weights, &format!("{}.conv1d.weight", p)),
                norm: norm::RMSNormGated {
                    weight: get_weight(&weights, &format!("{}.norm.weight", p)),
                    eps: args.rms_norm_eps,
                },
                dt_bias: get_weight(&weights, &format!("{}.dt_bias", p)),
                a_log: get_weight(&weights, &format!("{}.A_log", p)),
                num_v_heads: args.linear_num_value_heads,
                num_k_heads: args.linear_num_key_heads,
                head_k_dim: args.linear_key_head_dim,
                head_v_dim: args.linear_value_head_dim,
                key_dim: args.key_dim(),
                value_dim: args.value_dim(),
                conv_kernel_size: args.linear_conv_kernel_dim,
                conv_dim: args.conv_dim(),
            })
        } else {
            let p = format!("{}.self_attn", prefix);
            AttentionLayer::Full(Attention {
                q_proj: load_qlinear(&weights, &format!("{}.q_proj", p), bits, group_size),
                k_proj: load_qlinear(&weights, &format!("{}.k_proj", p), bits, group_size),
                v_proj: load_qlinear(&weights, &format!("{}.v_proj", p), bits, group_size),
                o_proj: load_qlinear(&weights, &format!("{}.o_proj", p), bits, group_size),
                q_norm: RMSNorm {
                    weight: get_weight(&weights, &format!("{}.q_norm.weight", p)),
                    eps: args.rms_norm_eps,
                },
                k_norm: RMSNorm {
                    weight: get_weight(&weights, &format!("{}.k_norm.weight", p)),
                    eps: args.rms_norm_eps,
                },
                num_heads: args.num_attention_heads,
                num_kv_heads: args.num_key_value_heads,
                head_dim: args.head_dim,
                rope_dims: args.rope_dims(),
                rope_theta: args.rope_theta as f32,
                scale: (args.head_dim as f32).powf(-0.5),
            })
        };

        let mlp_prefix = format!("{}.mlp", prefix);
        let mlp = MoeVariant::Qwen(SparseMoeBlock {
            gate: load_qlinear(&weights, &format!("{}.gate", mlp_prefix), bits, group_size),
            shared_expert: MLP {
                gate_proj: load_qlinear(&weights, &format!("{}.shared_expert.gate_proj", mlp_prefix), bits, group_size),
                up_proj: load_qlinear(&weights, &format!("{}.shared_expert.up_proj", mlp_prefix), bits, group_size),
                down_proj: load_qlinear(&weights, &format!("{}.shared_expert.down_proj", mlp_prefix), bits, group_size),
            },
            shared_expert_gate: load_qlinear(&weights, &format!("{}.shared_expert_gate", mlp_prefix), bits, group_size),
            top_k: args.experts_per_tok(),
            norm_topk_prob: args.norm_topk_prob,
            layer_idx: i,
            bits,
            group_size,
        });

        layers.push(DecoderLayer {
            attention,
            input_layernorm: input_ln,
            post_attention_layernorm: post_ln,
            mlp,
            pre_feedforward_layernorm: None,
            post_feedforward_layernorm: None,
            post_feedforward_layernorm_1: None,
            post_feedforward_layernorm_2: None,
            pre_feedforward_layernorm_2: None,
            dense_mlp: None,
            layer_scalar: None,
        });

        if (i + 1) % 10 == 0 || i == args.num_hidden_layers - 1 {
            eprintln!("  Built layer {}/{}", i + 1, args.num_hidden_layers);
        }
    }

    let final_norm = RMSNorm {
        weight: get_weight(&weights, "model.norm.weight"),
        eps: args.rms_norm_eps,
    };
    let lm_head = load_qlinear(&weights, "lm_head", bits, group_size);

    Ok(Model {
        model: TextModel {
            embed_tokens_weight: get_weight(&weights, "model.embed_tokens.weight"),
            embed_tokens_scales: Some(get_weight(&weights, "model.embed_tokens.scales")),
            embed_tokens_biases: Some(get_weight(&weights, "model.embed_tokens.biases")),
            embed_bits: bits,
            embed_group_size: group_size,
            layers,
            norm: final_norm,
            full_attention_interval: args.full_attention_interval,
            model_type: ModelType::Qwen,
            embed_scale: None,
        },
        lm_head,
        tie_word_embeddings: args.tie_word_embeddings,
        head_dim: args.head_dim,
        final_logit_softcapping: None,
    })
}

fn load_gemma4_model(split_path: &Path, args: &TextModelArgs, quant: Option<&crate::config::QuantizationConfig>) -> anyhow::Result<Model> {
    eprintln!("Loading resident weights (Gemma4)...");
    let resident_path = split_path.join("resident/resident.safetensors");
    let weights = load_safetensors_map(&resident_path)?;

    eprintln!(
        "Loaded {} resident tensors ({:.2} GB)",
        weights.len(),
        weights.values().map(|a| a.nbytes()).sum::<usize>() as f64 / 1e9
    );

    let bits = quant.map(|q| q.bits as i32).unwrap_or(8);
    let group_size = quant.map(|q| q.group_size as i32).unwrap_or(32);
    eprintln!("  Quantization: {}-bit, group_size={}", bits, group_size);

    let layer_types = args.layer_types.as_ref().unwrap();
    let global_head_dim = args.global_head_dim.unwrap_or(args.head_dim);
    let num_global_kv_heads = args.num_global_key_value_heads.unwrap_or(args.num_key_value_heads);

    let mut layers = Vec::with_capacity(args.num_hidden_layers);
    for i in 0..args.num_hidden_layers {
        let prefix = format!("language_model.layers.{}", i);
        let is_full = layer_types[i] == "full_attention";
        let use_k_eq_v = args.attention_k_eq_v && is_full;

        let input_ln = RMSNorm {
            weight: get_weight(&weights, &format!("{}.input_layernorm.weight", prefix)),
            eps: args.rms_norm_eps,
        };
        let post_attn_ln = RMSNorm {
            weight: get_weight(&weights, &format!("{}.post_attention_layernorm.weight", prefix)),
            eps: args.rms_norm_eps,
        };

        // Attention
        let p = format!("{}.self_attn", prefix);
        let head_dim = if is_full { global_head_dim } else { args.head_dim };
        let num_kv_heads = if use_k_eq_v { num_global_kv_heads } else { args.num_key_value_heads };
        let (rope_dims, rope_theta) = args.gemma4_rope_config(is_full);

        let v_proj = if use_k_eq_v {
            None
        } else {
            Some(load_qlinear_flex(&weights, &format!("{}.v_proj", p), bits, group_size))
        };

        let attention = AttentionLayer::Gemma4(Gemma4Attention {
            q_proj: load_qlinear_flex(&weights, &format!("{}.q_proj", p), bits, group_size),
            k_proj: load_qlinear_flex(&weights, &format!("{}.k_proj", p), bits, group_size),
            v_proj,
            o_proj: load_qlinear_flex(&weights, &format!("{}.o_proj", p), bits, group_size),
            q_norm: RMSNorm {
                weight: get_weight(&weights, &format!("{}.q_norm.weight", p)),
                eps: args.rms_norm_eps,
            },
            k_norm: RMSNorm {
                weight: get_weight(&weights, &format!("{}.k_norm.weight", p)),
                eps: args.rms_norm_eps,
            },
            v_norm: norm::RMSNormNoScale { eps: args.rms_norm_eps },
            num_heads: args.num_attention_heads,
            num_kv_heads,
            head_dim,
            rope_dims,
            rope_theta,
            use_k_eq_v,
        });

        // Feedforward norms
        let pre_ffn_ln = RMSNorm {
            weight: get_weight(&weights, &format!("{}.pre_feedforward_layernorm.weight", prefix)),
            eps: args.rms_norm_eps,
        };
        let post_ffn_ln = RMSNorm {
            weight: get_weight(&weights, &format!("{}.post_feedforward_layernorm.weight", prefix)),
            eps: args.rms_norm_eps,
        };
        let post_ffn_ln_1 = RMSNorm {
            weight: get_weight(&weights, &format!("{}.post_feedforward_layernorm_1.weight", prefix)),
            eps: args.rms_norm_eps,
        };
        let post_ffn_ln_2 = RMSNorm {
            weight: get_weight(&weights, &format!("{}.post_feedforward_layernorm_2.weight", prefix)),
            eps: args.rms_norm_eps,
        };
        let pre_ffn_ln_2 = RMSNorm {
            weight: get_weight(&weights, &format!("{}.pre_feedforward_layernorm_2.weight", prefix)),
            eps: args.rms_norm_eps,
        };

        // Dense MLP (runs in parallel with MoE)
        let mlp_prefix = format!("{}.mlp", prefix);
        let dense_mlp = GeLUMLP {
            gate_proj: load_qlinear_flex(&weights, &format!("{}.gate_proj", mlp_prefix), bits, group_size),
            up_proj: load_qlinear_flex(&weights, &format!("{}.up_proj", mlp_prefix), bits, group_size),
            down_proj: load_qlinear_flex(&weights, &format!("{}.down_proj", mlp_prefix), bits, group_size),
        };

        // MoE router
        let router_prefix = format!("{}.router", prefix);
        let moe = MoeVariant::Gemma4(Gemma4MoeBlock {
            router_proj: load_qlinear_flex(&weights, &format!("{}.proj", router_prefix), bits, group_size),
            router_scale: get_weight(&weights, &format!("{}.scale", router_prefix)),
            per_expert_scale: get_weight(&weights, &format!("{}.per_expert_scale", router_prefix)),
            root_size: (args.hidden_size as f32).powf(-0.5),
            rms_norm_eps: args.rms_norm_eps,
            top_k: args.experts_per_tok(),
            layer_idx: i,
            bits,
            group_size,
        });

        // Layer scalar
        let layer_scalar = weights.get(&format!("{}.layer_scalar", prefix)).cloned();

        layers.push(DecoderLayer {
            attention,
            input_layernorm: input_ln,
            post_attention_layernorm: post_attn_ln,
            mlp: moe,
            pre_feedforward_layernorm: Some(pre_ffn_ln),
            post_feedforward_layernorm: Some(post_ffn_ln),
            post_feedforward_layernorm_1: Some(post_ffn_ln_1),
            post_feedforward_layernorm_2: Some(post_ffn_ln_2),
            pre_feedforward_layernorm_2: Some(pre_ffn_ln_2),
            dense_mlp: Some(dense_mlp),
            layer_scalar,
        });

        if (i + 1) % 10 == 0 || i == args.num_hidden_layers - 1 {
            eprintln!("  Built layer {}/{}", i + 1, args.num_hidden_layers);
        }
    }

    let final_norm = RMSNorm {
        weight: get_weight(&weights, "language_model.norm.weight"),
        eps: args.rms_norm_eps,
    };

    // Embedding is NOT quantized for Gemma4 — just bf16 weight
    let embed_weight = get_weight(&weights, "language_model.embed_tokens.weight");

    // Dummy lm_head (not used when tie_word_embeddings=true)
    let dummy_lm_head = QuantizedLinear {
        weight: Array::from_f32(0.0),
        scales: Array::from_f32(0.0),
        biases: Array::from_f32(0.0),
        bits,
        group_size,
    };

    Ok(Model {
        model: TextModel {
            embed_tokens_weight: embed_weight,
            embed_tokens_scales: None,
            embed_tokens_biases: None,
            embed_bits: bits,
            embed_group_size: group_size,
            layers,
            norm: final_norm,
            full_attention_interval: 1, // not used for Gemma4
            model_type: ModelType::Gemma4,
            embed_scale: Some((args.hidden_size as f32).sqrt()),
        },
        lm_head: dummy_lm_head,
        tie_word_embeddings: args.tie_word_embeddings,
        head_dim: args.head_dim,
        final_logit_softcapping: args.final_logit_softcapping,
    })
}

// --- Helpers ---

fn load_safetensors_map(path: &Path) -> anyhow::Result<HashMap<String, Array>> {
    let map = Array::load_safetensors(path)
        .map_err(|e| anyhow::anyhow!("failed to load {}: {:?}", path.display(), e))?;
    Ok(map)
}

fn get_weight(weights: &HashMap<String, Array>, key: &str) -> Array {
    weights
        .get(key)
        .unwrap_or_else(|| panic!("missing weight: {}", key))
        .clone()
}

fn load_qlinear(
    weights: &HashMap<String, Array>,
    prefix: &str,
    bits: i32,
    group_size: i32,
) -> QuantizedLinear {
    QuantizedLinear {
        weight: get_weight(weights, &format!("{}.weight", prefix)),
        scales: get_weight(weights, &format!("{}.scales", prefix)),
        biases: get_weight(weights, &format!("{}.biases", prefix)),
        bits,
        group_size,
    }
}

/// Load quantized linear with flexible naming (Qwen: .scales/.biases, Gemma4: .weight_scales/.weight_biases)
fn load_qlinear_flex(
    weights: &HashMap<String, Array>,
    prefix: &str,
    bits: i32,
    group_size: i32,
) -> QuantizedLinear {
    let weight = get_weight(weights, &format!("{}.weight", prefix));
    let scales = weights.get(&format!("{}.scales", prefix))
        .or_else(|| weights.get(&format!("{}.weight_scales", prefix)))
        .unwrap_or_else(|| panic!("missing scales for: {}", prefix))
        .clone();
    let biases = weights.get(&format!("{}.biases", prefix))
        .or_else(|| weights.get(&format!("{}.weight_biases", prefix)))
        .unwrap_or_else(|| panic!("missing biases for: {}", prefix))
        .clone();
    QuantizedLinear { weight, scales, biases, bits, group_size }
}

fn create_attention_mask(
    hidden: &Array,
    cache_offset: usize,
) -> Result<Option<Array>, Exception> {
    let seq_len = hidden.dim(1) as usize;
    if seq_len <= 1 {
        return Ok(None);
    }
    let total_len = cache_offset + seq_len;
    let rows = Array::from_iter(
        (cache_offset as i32)..(total_len as i32),
        &[seq_len as i32, 1],
    );
    let cols = Array::from_iter(0..(total_len as i32), &[1, total_len as i32]);
    let mask = rows.ge(&cols)?;
    let zero = Array::from_f32(0.0);
    let neg_inf = Array::from_f32(f32::NEG_INFINITY);
    let additive = mlx_rs::ops::r#where(&mask, &zero, &neg_inf)?;
    let additive = additive.reshape(&[1, 1, seq_len as i32, total_len as i32])?;
    let additive = additive.as_dtype(hidden.dtype())?;
    Ok(Some(additive))
}

/// Create a sliding window causal mask for Gemma4 sliding attention.
fn create_sliding_mask(
    hidden: &Array,
    cache_offset: usize,
    _has_sliding_window: bool,
) -> Result<Option<Array>, Exception> {
    // For now, use standard causal mask (sliding window is only important for long sequences)
    create_attention_mask(hidden, cache_offset)
}

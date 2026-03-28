pub mod attention;
pub mod gated_delta;
pub mod mlp;
pub mod moe;
pub mod norm;

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::error::Exception;
use mlx_rs::Array;

use crate::cache::{ArraysCache, Cache, KVCache};
use crate::config::TextModelArgs;
use crate::memory::ExpertMemoryManager;
use attention::Attention;
use gated_delta::GatedDeltaNet;
use mlp::{QuantizedLinear, MLP};
use moe::SparseMoeBlock;
use norm::RMSNorm;

pub enum AttentionLayer {
    Linear(GatedDeltaNet),
    Full(Attention),
}

pub struct DecoderLayer {
    pub attention: AttentionLayer,
    pub input_layernorm: RMSNorm,
    pub post_attention_layernorm: RMSNorm,
    pub mlp: SparseMoeBlock,
}

impl DecoderLayer {
    pub fn is_linear(&self) -> bool {
        matches!(&self.attention, AttentionLayer::Linear(_))
    }

    pub fn forward(
        &mut self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut Cache,
        mem: &ExpertMemoryManager,
    ) -> Result<Array, Exception> {
        let normed = self.input_layernorm.forward(x)?;
        let attn_out = match &mut self.attention {
            AttentionLayer::Linear(gdn) => {
                gdn.forward(&normed, mask, cache.as_arrays_mut())?
            }
            AttentionLayer::Full(attn) => {
                attn.forward(&normed, mask, cache.as_kv_mut())?
            }
        };
        let h = x + &attn_out;
        let normed = self.post_attention_layernorm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed, mem)?;
        Ok(&h + &mlp_out)
    }
}

pub struct TextModel {
    pub embed_tokens_weight: Array,
    pub embed_tokens_scales: Array,
    pub embed_tokens_biases: Array,
    pub embed_bits: i32,
    pub embed_group_size: i32,
    pub layers: Vec<DecoderLayer>,
    pub norm: RMSNorm,
    pub full_attention_interval: usize,
}

impl TextModel {
    pub fn forward(
        &mut self,
        input_ids: &Array,
        cache: &mut [Cache],
        mem: &ExpertMemoryManager,
    ) -> Result<Array, Exception> {
        let flat_ids = input_ids.flatten(None, None)?;
        let w = mlx_rs::ops::indexing::take_axis(&self.embed_tokens_weight, &flat_ids, 0)?;
        let s = mlx_rs::ops::indexing::take_axis(&self.embed_tokens_scales, &flat_ids, 0)?;
        let b = mlx_rs::ops::indexing::take_axis(&self.embed_tokens_biases, &flat_ids, 0)?;
        let hidden = mlx_rs::ops::dequantize(
            &w, &s, &b, Some(self.embed_group_size), Some(self.embed_bits),
        )?;
        let shape = input_ids.shape();
        let hidden = hidden.reshape(&[shape[0], shape[1], -1])?;

        let fa_idx = self.full_attention_interval - 1;
        let fa_offset = cache[fa_idx].kv_offset();
        let fa_mask = create_attention_mask(&hidden, fa_offset)?;

        let mut h = hidden;
        for (i, (layer, c)) in self.layers.iter_mut().zip(cache.iter_mut()).enumerate() {
            let mask = if layer.is_linear() {
                None
            } else {
                fa_mask.as_ref()
            };
            h = layer.forward(&h, mask, c, mem)?;
            mlx_rs::transforms::eval(std::iter::once(&h))?;
        }

        self.norm.forward(&h)
    }
}

pub struct Model {
    pub model: TextModel,
    pub lm_head: QuantizedLinear,
    pub tie_word_embeddings: bool,
}

impl Model {
    pub fn forward(
        &mut self,
        input_ids: &Array,
        cache: &mut [Cache],
        mem: &ExpertMemoryManager,
    ) -> Result<Array, Exception> {
        let out = self.model.forward(input_ids, cache, mem)?;
        if self.tie_word_embeddings {
            mlx_rs::ops::quantized_matmul(
                &out,
                &self.model.embed_tokens_weight,
                &self.model.embed_tokens_scales,
                &self.model.embed_tokens_biases,
                Some(true),
                Some(self.model.embed_group_size),
                Some(self.model.embed_bits),
            )
        } else {
            self.lm_head.forward(&out)
        }
    }

    pub fn make_cache(&self) -> Vec<Cache> {
        self.model
            .layers
            .iter()
            .map(|layer| {
                if layer.is_linear() {
                    Cache::Arrays(ArraysCache::new(2))
                } else {
                    Cache::KV(KVCache::new())
                }
            })
            .collect()
    }
}

// --- Weight loading ---

pub fn load_model(split_path: &Path, args: &TextModelArgs) -> anyhow::Result<Model> {
    eprintln!("Loading resident weights...");
    let resident_path = split_path.join("resident/resident.safetensors");
    let weights = load_safetensors_map(&resident_path)?;

    eprintln!(
        "Loaded {} resident tensors ({:.2} GB)",
        weights.len(),
        weights.values().map(|a| a.nbytes()).sum::<usize>() as f64 / 1e9
    );

    // Expert weights are NOT loaded here — they're mmap'd by ExpertMemoryManager
    // and extracted on-demand during forward passes (~27 MB per layer vs 34.6 GB)

    let bits = 8i32;
    let group_size = 32i32;

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
        let mlp = SparseMoeBlock {
            gate: load_qlinear(&weights, &format!("{}.gate", mlp_prefix), bits, group_size),
            shared_expert: MLP {
                gate_proj: load_qlinear(&weights, &format!("{}.shared_expert.gate_proj", mlp_prefix), bits, group_size),
                up_proj: load_qlinear(&weights, &format!("{}.shared_expert.up_proj", mlp_prefix), bits, group_size),
                down_proj: load_qlinear(&weights, &format!("{}.shared_expert.down_proj", mlp_prefix), bits, group_size),
            },
            shared_expert_gate: load_qlinear(&weights, &format!("{}.shared_expert_gate", mlp_prefix), bits, group_size),
            top_k: args.num_experts_per_tok,
            norm_topk_prob: args.norm_topk_prob,
            layer_idx: i,
            bits,
            group_size,
        };

        layers.push(DecoderLayer {
            attention,
            input_layernorm: input_ln,
            post_attention_layernorm: post_ln,
            mlp,
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
            embed_tokens_scales: get_weight(&weights, "model.embed_tokens.scales"),
            embed_tokens_biases: get_weight(&weights, "model.embed_tokens.biases"),
            embed_bits: bits,
            embed_group_size: group_size,
            layers,
            norm: final_norm,
            full_attention_interval: args.full_attention_interval,
        },
        lm_head,
        tie_word_embeddings: args.tie_word_embeddings,
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

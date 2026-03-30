use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::OnceLock;
use std::time::Instant;

use mlx_rs::error::Exception;
use mlx_rs::Array;

use crate::ffi;
use crate::memory::{ExpertMemoryManager, SingleExpertTensors};
use crate::model::mlp::{QuantizedLinear, MLP};
use crate::model::norm::RMSNorm;
use crate::perf::PerfStats;

/// Tracks gate-reuse prediction accuracy: how well layer L's gate input
/// predicts layer L+1's routing when run through L+1's router.
pub struct TransitionProfiler {
    num_layers: usize,
    gate_total: usize,
    gate_hits: usize,
    gate_per_layer_total: Vec<usize>,
    gate_per_layer_hits: Vec<usize>,
    /// Pending prediction from previous layer (layer_idx, predicted_experts)
    pub pending_prediction: Option<(usize, Vec<i32>)>,
}

impl TransitionProfiler {
    pub fn new(num_layers: usize) -> Self {
        Self {
            num_layers,
            gate_total: 0,
            gate_hits: 0,
            gate_per_layer_total: vec![0; num_layers],
            gate_per_layer_hits: vec![0; num_layers],
            pending_prediction: None,
        }
    }

    /// Compare predicted experts (from gate reuse) against actual routing.
    pub fn record_gate_reuse(&mut self, layer: usize, predicted: &[i32], actual: &[i32]) {
        for &e in actual {
            self.gate_total += 1;
            self.gate_per_layer_total[layer] += 1;
            if predicted.contains(&e) {
                self.gate_hits += 1;
                self.gate_per_layer_hits[layer] += 1;
            }
        }
    }

    /// End of token: clear pending prediction.
    pub fn end_token(&mut self) {
        self.pending_prediction = None;
    }

    /// Report prediction accuracy.
    pub fn report(&self) {
        if self.gate_total > 0 {
            eprintln!("\n=== Gate-Reuse Prediction (pre-MoE + next LN, top-12) ===");
            eprintln!(
                "  Overall: {:.1}% ({}/{})",
                self.gate_hits as f64 / self.gate_total as f64 * 100.0,
                self.gate_hits,
                self.gate_total
            );
            for i in 0..self.num_layers {
                let t = self.gate_per_layer_total[i];
                if t > 0 {
                    let h = self.gate_per_layer_hits[i];
                    eprintln!(
                        "  Layer {:>2}: {:.1}% ({}/{})",
                        i,
                        h as f64 / t as f64 * 100.0,
                        h,
                        t
                    );
                }
            }
        }
    }
}

/// If true, use zero-copy mmap + per-expert quantized_matmul instead of gather_qmm.
pub const USE_ZEROCOPY: bool = true;

/// Cached check for NOREACTIVE env var (avoids 40 env::var lookups per token).
static NOREACTIVE: OnceLock<bool> = OnceLock::new();

/// UMA-native sparse MoE block.
/// Expert weights are loaded on-demand from mmap'd safetensors (not held in memory).
pub struct SparseMoeBlock {
    pub gate: QuantizedLinear,
    pub shared_expert: MLP,
    pub shared_expert_gate: QuantizedLinear,
    pub top_k: usize,
    pub norm_topk_prob: bool,
    pub layer_idx: usize,
    pub bits: i32,
    pub group_size: i32,
}

impl SparseMoeBlock {
    pub fn forward(
        &self,
        x: &Array,
        mem: &ExpertMemoryManager,
        perf: &PerfStats,
        next_layer_gate: Option<(&QuantizedLinear, &RMSNorm)>,
        sync_preload: bool,
        tp: Option<&RefCell<TransitionProfiler>>,
    ) -> Result<Array, Exception> {
        let k = self.top_k as i32;

        // 1. Router
        let gates = self.gate.forward(x)?;
        let gates = mlx_rs::ops::softmax_axis(&gates, -1, Some(true))?;

        let inds_full = mlx_rs::ops::argpartition_axis(&gates, -k, -1)?;
        let num_experts_total = inds_full.dim(inds_full.ndim() as i32 - 1);
        let split_at = num_experts_total - k;
        let parts = mlx_rs::ops::split_sections(&inds_full, &[split_at], Some(-1))?;
        let inds = parts[1].as_dtype(mlx_rs::Dtype::Int32)?;
        let scores = mlx_rs::ops::indexing::take_along_axis(&gates, &inds, Some(-1))?;
        let scores = if self.norm_topk_prob {
            let s = mlx_rs::ops::sum_axis(&scores, -1, Some(true))?;
            &scores / &s
        } else {
            scores
        };

        // 2. Shared expert (independent path)
        let shared_y = self.shared_expert.forward(x)?;
        let shared_gate = mlx_rs::ops::sigmoid(&self.shared_expert_gate.forward(x)?)?;
        let shared_y = &shared_gate * &shared_y;

        // 3. Compute routing indices + scores (lazy, will batch with prediction)
        let flat_idx = inds.reshape(&[-1])?;
        let scores_f32 = scores.as_dtype(mlx_rs::Dtype::Float32)?;

        let seq_len = x.dim(1);
        let is_decode = seq_len == 1;

        // Speculative prediction for next layer (lazy graph, batched with routing eval)
        let pred_flat = if is_decode {
            if let Some((next_gate, next_ln)) = next_layer_gate {
                let pred_k = 12i32;
                let pred_normed = next_ln.forward(x)?;
                let pred_logits = next_gate.forward(&pred_normed)?;
                let pred_inds = mlx_rs::ops::argpartition_axis(&pred_logits, -pred_k, -1)?;
                let pred_num = pred_inds.dim(pred_inds.ndim() as i32 - 1);
                let pred_split = pred_num - pred_k;
                let pred_parts =
                    mlx_rs::ops::split_sections(&pred_inds, &[pred_split], Some(-1))?;
                let pred_topk = pred_parts[1].as_dtype(mlx_rs::Dtype::Int32)?;
                Some(pred_topk.reshape(&[-1])?)
            } else {
                None
            }
        } else {
            None
        };

        // Single GPU sync: routing + scores + prediction (zero extra sync overhead)
        let _t = Instant::now();
        if let Some(ref pf) = pred_flat {
            mlx_rs::transforms::eval([&flat_idx, &scores_f32, pf])?;
        } else {
            mlx_rs::transforms::eval([&flat_idx, &scores_f32])?;
        }
        perf.acc(&perf.moe_routing_eval, _t.elapsed());
        let flat_data: &[i32] = flat_idx.as_slice();
        let scores_data: &[f32] = scores_f32.as_slice();

        // Find unique expert indices and build remap table
        let _t = Instant::now();
        let mut unique: Vec<i32> = flat_data.to_vec();
        unique.sort();
        unique.dedup();
        let remap: HashMap<i32, i32> = unique
            .iter()
            .enumerate()
            .map(|(i, &orig)| (orig, i as i32))
            .collect();

        // Blocking prefetch: parallel pread all experts to warm page cache.
        // Prevents page faults during eval — pread at contiguous MB granularity
        // is far more efficient than 16KB page fault traps.
        // Skip if NOREACTIVE=1 (for A/B testing).
        if is_decode && !*NOREACTIVE.get_or_init(|| std::env::var("NOREACTIVE").is_ok()) {
            mem.pread_experts_sync(self.layer_idx, &unique);
        }

        // Pre-compute per-expert score weights (for zero-copy path)
        let mut weight_map: HashMap<i32, f32> = HashMap::new();
        for (i, &idx) in flat_data.iter().enumerate() {
            *weight_map.entry(idx).or_insert(0.0) += scores_data[i];
        }
        perf.acc(&perf.routing_cpu, _t.elapsed());

        // Check pending prediction from previous layer (gate-reuse accuracy)
        if let Some(tp_ref) = tp {
            let mut tp_mut = tp_ref.borrow_mut();
            if let Some((pred_layer, predicted)) = tp_mut.pending_prediction.take() {
                if pred_layer == self.layer_idx {
                    tp_mut.record_gate_reuse(self.layer_idx, &predicted, &unique);
                }
            }
        }

        // Record prediction for pipeline prefetch + fire F_RDADVISE
        if let Some(ref pf) = pred_flat {
            let pred_data: &[i32] = pf.as_slice();
            let mut pred_unique: Vec<i32> = pred_data.to_vec();
            pred_unique.sort();
            pred_unique.dedup();
            let next_layer = self.layer_idx + 1;
            // Prediction recorded — speculative prefetch fires from mod.rs
            // during eval (after blocking pread is done) to avoid SSD contention.
            if let Some(tp_ref) = tp {
                tp_ref.borrow_mut().pending_prediction = Some((next_layer, pred_unique));
            }
        }

        if USE_ZEROCOPY {
            let _t = Instant::now();

            // Use preloaded experts (from async_eval pipeline) or fall back to zerocopy
            let preloaded = mem.take_preloaded(self.layer_idx);
            let experts: Vec<SingleExpertTensors> = if let Some((_pre_idx, pre_batch)) = preloaded {
                // Preloaded from hybrid buffer — fault-free.
                // Handle prediction misses via hybrid pread into spare buffer.
                let mut pre_opts: Vec<Option<SingleExpertTensors>> =
                    pre_batch.into_iter().map(Some).collect();
                let mut result: Vec<Option<SingleExpertTensors>> = Vec::with_capacity(unique.len());
                let mut misses: Vec<(usize, i32)> = Vec::new();
                for (i, &eidx) in unique.iter().enumerate() {
                    if let Some(pos) = _pre_idx.iter().position(|&e| e == eidx) {
                        if let Some(expert) = pre_opts[pos].take() {
                            result.push(Some(expert));
                            continue;
                        }
                    }
                    misses.push((i, eidx));
                    result.push(None);
                }
                if !misses.is_empty() {
                    let miss_indices: Vec<i32> = misses.iter().map(|(_, e)| *e).collect();
                    let miss_experts = mem.extract_experts_hybrid(self.layer_idx, &miss_indices);
                    let mut miss_iter = miss_experts.into_iter();
                    for (i, _) in &misses {
                        result[*i] = Some(miss_iter.next().unwrap());
                    }
                }
                result.into_iter().map(|o| o.unwrap()).collect()
            } else if sync_preload && is_decode {
                // Pipeline mode, no preloaded data (layer 0): sync hybrid extract
                mem.extract_experts_hybrid(self.layer_idx, &unique)
            } else {
                // Prefill or non-pipeline decode: mmap zerocopy
                let mut v = Vec::with_capacity(unique.len());
                for &eidx in &unique {
                    v.push(mem.extract_expert_zerocopy(self.layer_idx, eidx));
                }
                v
            };

            let mut y_accum: Option<Array> = None;

            for (idx, &eidx) in unique.iter().enumerate() {
                let expert = &experts[idx];

                // Expert MLP: gate_proj → silu, up_proj, element-wise multiply, down_proj
                let gate_out = mlx_rs::ops::quantized_matmul(
                    x,
                    &expert.gate_weight,
                    &expert.gate_scales,
                    &expert.gate_biases,
                    true,
                    self.group_size,
                    self.bits,
                )?;
                let up_out = mlx_rs::ops::quantized_matmul(
                    x,
                    &expert.up_weight,
                    &expert.up_scales,
                    &expert.up_biases,
                    true,
                    self.group_size,
                    self.bits,
                )?;
                let act = &mlx_rs::nn::silu(&gate_out)? * &up_out;
                let down_out = mlx_rs::ops::quantized_matmul(
                    &act,
                    &expert.down_weight,
                    &expert.down_scales,
                    &expert.down_biases,
                    true,
                    self.group_size,
                    self.bits,
                )?;

                if is_decode {
                    // Decode: scalar weight per expert (single position)
                    let total_weight = weight_map.get(&eidx).copied().unwrap_or(0.0);
                    if total_weight == 0.0 {
                        continue;
                    }
                    let scale = Array::from_f32(total_weight).as_dtype(x.dtype())?;
                    let weighted = &down_out * &scale;
                    y_accum = Some(match y_accum {
                        None => weighted,
                        Some(acc) => &acc + &weighted,
                    });
                } else {
                    // Prefill: per-position weighting via MLX ops
                    let eidx_arr = Array::from_int(eidx);
                    let mask = inds.eq(&eidx_arr)?;
                    let mask_f = mask.as_dtype(scores.dtype())?;
                    let per_pos_weight =
                        mlx_rs::ops::sum_axis(&(&scores * &mask_f), -1, Some(true))?;
                    let weighted = &down_out * &per_pos_weight;
                    y_accum = Some(match y_accum {
                        None => weighted,
                        Some(acc) => &acc + &weighted,
                    });
                }
            }
            perf.acc(&perf.extract_experts, _t.elapsed());

            let y = y_accum.unwrap_or_else(|| Array::zeros::<f32>(&x.shape()).unwrap());
            Ok(&y + &shared_y)
        } else {
            // Original path: pread + scatter + gather_qmm

            // 4. Extract only the needed experts (~27 MB for 8 experts)
            let _t = Instant::now();
            let experts = mem.extract_experts(self.layer_idx, &unique);
            perf.acc(&perf.extract_experts, _t.elapsed());

            // Speculative prefetch: pre-warm next layer's pages via F_RDADVISE
            mem.prefetch_next_layer(self.layer_idx, &unique);

            // 5. Remap flat indices from [0-255] to [0-num_unique)
            let remapped: Vec<i32> = flat_data.iter().map(|&idx| remap[&idx]).collect();
            let remapped_idx = Array::from_slice(&remapped, &[flat_data.len() as i32]);

            // 6. Sort remapped indices for gather_qmm
            let x_exp = mlx_rs::ops::expand_dims_axes(x, &[-2, -3])?;
            let order = mlx_rs::ops::argsort(&remapped_idx)?;
            let inv_order = mlx_rs::ops::argsort(&order)?;
            let x_flat = mlx_rs::ops::flatten(&x_exp, Some(0), Some(-3))?;
            let div_k = mlx_rs::ops::floor_divide(&order, &Array::from_int(k))?;
            let x_sorted = mlx_rs::ops::indexing::take_axis(&x_flat, &div_k, 0)?;
            let idx_sorted = mlx_rs::ops::indexing::take_axis(&remapped_idx, &order, 0)?;

            let _t = Instant::now();
            mlx_rs::transforms::eval([&x_sorted, &idx_sorted])?;
            perf.acc(&perf.moe_sort_eval, _t.elapsed());

            // 7. gather_qmm triad on compact expert arrays
            let x_gate = ffi::gather_qmm(
                &x_sorted,
                &experts.gate_weight,
                &experts.gate_scales,
                &experts.gate_biases,
                &idx_sorted,
                true,
                self.group_size,
                self.bits,
                true,
            )?;
            let x_up = ffi::gather_qmm(
                &x_sorted,
                &experts.up_weight,
                &experts.up_scales,
                &experts.up_biases,
                &idx_sorted,
                true,
                self.group_size,
                self.bits,
                true,
            )?;
            let x_act = &mlx_rs::nn::silu(&x_gate)? * &x_up;
            let x_down = ffi::gather_qmm(
                &x_act,
                &experts.down_weight,
                &experts.down_scales,
                &experts.down_biases,
                &idx_sorted,
                true,
                self.group_size,
                self.bits,
                true,
            )?;

            // 8. Unsort + weighted sum
            let x_down = mlx_rs::ops::indexing::take_axis(&x_down, &inv_order, 0)?;
            let target_shape = inds.shape().to_vec();
            let x_down = mlx_rs::ops::unflatten(&x_down, 0, &target_shape)?;
            let x_down = mlx_rs::ops::squeeze_axes(&x_down, &[-2])?;

            let scores_exp = mlx_rs::ops::expand_dims(&scores, -1)?;
            let y = mlx_rs::ops::sum_axis(&(&x_down * &scores_exp), -2, Some(false))?;

            Ok(&y + &shared_y)
        }
    }
}

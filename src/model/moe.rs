use std::collections::HashMap;
use std::time::Instant;

use mlx_rs::error::Exception;
use mlx_rs::Array;

use crate::ffi;
use crate::memory::ExpertMemoryManager;
use crate::model::mlp::{QuantizedLinear, MLP};
use crate::perf::PerfStats;

/// If true, use zero-copy mmap + per-expert quantized_matmul instead of gather_qmm.
pub const USE_ZEROCOPY: bool = true;

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
    pub fn forward(&self, x: &Array, mem: &ExpertMemoryManager, perf: &PerfStats) -> Result<Array, Exception> {
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

        // 3. Eval routing indices → read to CPU for on-demand expert loading.
        // Single eval materializes the entire routing chain (gate + softmax + argpartition)
        // plus any pending lazy work from attention (GDN recurrent tail during decode).
        let flat_idx = inds.reshape(&[-1])?;
        let _t = Instant::now();
        mlx_rs::transforms::eval(std::iter::once(&flat_idx))?;
        perf.acc(&perf.moe_routing_eval, _t.elapsed());
        let flat_data: &[i32] = flat_idx.as_slice();

        // Find unique expert indices and build remap table
        let _t = Instant::now();
        let mut unique: Vec<i32> = flat_data.to_vec();
        unique.sort();
        unique.dedup();
        let remap: HashMap<i32, i32> = unique.iter().enumerate()
            .map(|(i, &orig)| (orig, i as i32))
            .collect();
        perf.acc(&perf.routing_cpu, _t.elapsed());

        // Record routing for cross-token predictor
        mem.predictor_record(self.layer_idx, &unique);

        if USE_ZEROCOPY {
            // Zero-copy path: per-expert quantized_matmul from mmap'd Metal buffers

            // 4. For each expert, create zero-copy arrays and compute matmuls
            let _t = Instant::now();

            // Build per-token score map: for each position, which experts and what weights?
            // inds shape: [batch, top_k], scores shape: [batch, top_k]
            // For decode (batch=1), inds is [1, 8] and flat_data is [8]
            let mut y_accum: Option<Array> = None;

            for (ei, &eidx) in unique.iter().enumerate() {
                let expert = mem.extract_expert_zerocopy(self.layer_idx, eidx);

                // Compute expert MLP: gate_proj → silu, up_proj, element-wise multiply, down_proj
                let gate_out = mlx_rs::ops::quantized_matmul(
                    x, &expert.gate_weight, &expert.gate_scales, &expert.gate_biases,
                    true, self.group_size, self.bits,
                )?;
                let up_out = mlx_rs::ops::quantized_matmul(
                    x, &expert.up_weight, &expert.up_scales, &expert.up_biases,
                    true, self.group_size, self.bits,
                )?;
                let act = &mlx_rs::nn::silu(&gate_out)? * &up_out;
                let down_out = mlx_rs::ops::quantized_matmul(
                    &act, &expert.down_weight, &expert.down_scales, &expert.down_biases,
                    true, self.group_size, self.bits,
                )?;

                // Weight by per-position routing scores for this expert.
                // For each position, find the score assigned to this expert.
                // flat_data has the expert indices per (batch, top_k) position.
                // Build a mask/weight for positions that selected this expert.
                let score_weights: Vec<f32> = flat_data.iter().enumerate()
                    .map(|(_pos, &idx)| if idx == eidx { 1.0 } else { 0.0 })
                    .collect();

                // Get the actual scores for positions that match.
                // scores is [1, top_k] in model dtype (bf16) — convert to f32 for accumulation.
                let scores_f32 = scores.as_dtype(mlx_rs::Dtype::Float32)?;
                mlx_rs::transforms::eval(std::iter::once(&scores_f32))?;
                let scores_data: &[f32] = scores_f32.as_slice();
                let total_weight: f32 = score_weights.iter().enumerate()
                    .map(|(i, &w)| if w > 0.0 { scores_data[i] } else { 0.0 })
                    .sum();

                if total_weight > 0.0 {
                    let scale = Array::from_f32(total_weight);
                    let weighted = &down_out * &scale;
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
                &x_sorted, &experts.gate_weight, &experts.gate_scales, &experts.gate_biases,
                &idx_sorted, true, self.group_size, self.bits, true,
            )?;
            let x_up = ffi::gather_qmm(
                &x_sorted, &experts.up_weight, &experts.up_scales, &experts.up_biases,
                &idx_sorted, true, self.group_size, self.bits, true,
            )?;
            let x_act = &mlx_rs::nn::silu(&x_gate)? * &x_up;
            let x_down = ffi::gather_qmm(
                &x_act, &experts.down_weight, &experts.down_scales, &experts.down_biases,
                &idx_sorted, true, self.group_size, self.bits, true,
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

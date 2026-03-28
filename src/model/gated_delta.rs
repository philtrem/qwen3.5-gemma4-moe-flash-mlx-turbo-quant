use mlx_rs::error::Exception;
use mlx_rs::{Array, Dtype};

use crate::cache::ArraysCache;
use crate::model::mlp::QuantizedLinear;
use crate::model::norm::RMSNormGated;

/// GatedDeltaNet — linear attention (30 of 40 layers).
/// Uses ops-based fallback (Path B) for the recurrent update.
pub struct GatedDeltaNet {
    pub in_proj_qkv: QuantizedLinear,
    pub in_proj_z: QuantizedLinear,
    pub in_proj_b: QuantizedLinear,
    pub in_proj_a: QuantizedLinear,
    pub out_proj: QuantizedLinear,
    pub conv1d_weight: Array,
    pub norm: RMSNormGated,
    pub dt_bias: Array,
    pub a_log: Array,
    pub num_v_heads: usize,
    pub num_k_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub conv_kernel_size: usize,
    pub conv_dim: usize,
}

impl GatedDeltaNet {
    pub fn forward(
        &self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut ArraysCache,
    ) -> Result<Array, Exception> {
        let b = x.dim(0);
        let s = x.dim(1);

        // 1. Project
        let qkv = self.in_proj_qkv.forward(x)?;
        let z = self.in_proj_z.forward(x)?
            .reshape(&[b, s, self.num_v_heads as i32, self.head_v_dim as i32])?;
        let b_proj = self.in_proj_b.forward(x)?;
        let a_proj = self.in_proj_a.forward(x)?;

        // 2. Conv1d with state
        let conv_input = if let Some(state) = cache.get(0) {
            mlx_rs::ops::concatenate_axis(&[state, &qkv], 1)?
        } else {
            let zeros = mlx_rs::ops::zeros::<f32>(&[
                b,
                (self.conv_kernel_size - 1) as i32,
                self.conv_dim as i32,
            ])?.as_dtype(x.dtype())?;
            mlx_rs::ops::concatenate_axis(&[&zeros, &qkv], 1)?
        };

        // Save conv state: last (kernel_size - 1) timesteps via narrow slice
        let total_t = conv_input.dim(1) as usize;
        let keep = self.conv_kernel_size - 1;
        // Use split_sections to slice: split at [total_t - keep] along axis 1
        let split_at = (total_t - keep) as i32;
        let slices = mlx_rs::ops::split_sections(&conv_input, &[split_at], Some(1))?;
        let new_conv_state = slices[1].clone();
        cache.set(0, new_conv_state);

        // Depthwise conv1d + silu
        let conv_out = self.depthwise_conv1d(&conv_input)?;
        let conv_out = mlx_rs::nn::silu(&conv_out)?;

        // 3. Split conv output into Q, K, V along last axis
        let kd = self.key_dim as i32;
        let kd2 = (self.key_dim * 2) as i32;
        let parts = mlx_rs::ops::split_sections(&conv_out, &[kd, kd2], Some(-1))?;
        let q = parts[0].reshape(&[b, s, self.num_k_heads as i32, self.head_k_dim as i32])?;
        let k = parts[1].reshape(&[b, s, self.num_k_heads as i32, self.head_k_dim as i32])?;
        let v = parts[2].reshape(&[b, s, self.num_v_heads as i32, self.head_v_dim as i32])?;

        // 4. RMS norm with scaling
        let inv_scale = (self.head_k_dim as f32).powf(-0.5);
        let norm_w = mlx_rs::ops::ones::<f32>(&[self.head_k_dim as i32])?.as_dtype(x.dtype())?;
        let q = mlx_rs::fast::rms_norm(&q, &norm_w, 1e-6)?;
        let inv2 = Array::from_f32(inv_scale * inv_scale).as_dtype(x.dtype())?;
        let q = &q * &inv2;
        let k = mlx_rs::fast::rms_norm(&k, &norm_w, 1e-6)?;
        let inv1 = Array::from_f32(inv_scale).as_dtype(x.dtype())?;
        let k = &k * &inv1;

        // 5. Compute gating
        let beta = mlx_rs::ops::sigmoid(&b_proj)?;
        let g = compute_g(&self.a_log, &a_proj, &self.dt_bias)?;

        // Eval inputs before sequential recurrent loop
        mlx_rs::transforms::eval([&q, &k, &v, &beta, &g])?;
        // 6. Recurrent update (Path B)
        let state = cache.get(1).cloned();
        let (out, new_state) = gated_delta_ops(&q, &k, &v, &g, &beta, state.as_ref(), mask)?;
        cache.set(1, new_state);

        // 7. Gated norm + output projection
        let out = self.norm.forward(&out, &z)?;
        let out = out.reshape(&[b, s, -1])?;
        self.out_proj.forward(&out)
    }

    /// Depthwise conv1d (manual implementation).
    fn depthwise_conv1d(&self, x: &Array) -> Result<Array, Exception> {
        let k = self.conv_kernel_size;
        // Weight: [conv_dim, kernel_size, 1] → squeeze last dim → [conv_dim, kernel_size]
        let w = mlx_rs::ops::squeeze_axes(&self.conv1d_weight, &[2])?;
        // Transpose to [kernel_size, conv_dim]
        let w = mlx_rs::ops::transpose(&w)?;

        let t_out = x.dim(1) - k as i32 + 1;
        let mut result = mlx_rs::ops::zeros::<f32>(&[x.dim(0), t_out, x.dim(2)])?
            .as_dtype(x.dtype())?;

        for i in 0..k {
            let start = i as i32;
            let indices: Vec<i32> = (start..start + t_out).collect();
            let x_slice = mlx_rs::ops::indexing::take_axis(
                x,
                &Array::from_slice(&indices, &[t_out]),
                1,
            )?;
            let w_i = mlx_rs::ops::indexing::take_axis(
                &w,
                &Array::from_slice(&[i as i32], &[1]),
                0,
            )?;
            result = &result + &(&x_slice * &w_i);
        }

        Ok(result)
    }
}

/// g = exp(-exp(A_log.f32) * softplus(a + dt_bias))
fn compute_g(a_log: &Array, a: &Array, dt_bias: &Array) -> Result<Array, Exception> {
    let orig_dtype = a.dtype();
    let a_log_f32 = a_log.as_dtype(Dtype::Float32)?;
    let a_f32 = a.as_dtype(Dtype::Float32)?;
    let dt_bias_f32 = dt_bias.as_dtype(Dtype::Float32)?;

    let exp_a_log = mlx_rs::ops::exp(&a_log_f32)?;
    let sum = &a_f32 + &dt_bias_f32;
    let sp = softplus(&sum)?;
    let neg_product = &(-&exp_a_log) * &sp;
    let g = mlx_rs::ops::exp(&neg_product)?;
    g.as_dtype(orig_dtype)
}

fn softplus(x: &Array) -> Result<Array, Exception> {
    let exp_x = mlx_rs::ops::exp(x)?;
    let one_plus = &exp_x + 1.0f32;
    mlx_rs::ops::log(&one_plus)
}

/// Ops-based recurrent update (Path B fallback).
fn gated_delta_ops(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: Option<&Array>,
    mask: Option<&Array>,
) -> Result<(Array, Array), Exception> {
    let b = q.dim(0);
    let t = q.dim(1) as usize;
    let hk = q.dim(2) as usize;
    let dk = q.dim(3);
    let hv = v.dim(2) as usize;
    let dv = v.dim(3);

    let mut state = match state {
        Some(s) => s.clone(),
        None => mlx_rs::ops::zeros::<f32>(&[b, hv as i32, dv, dk])?
            .as_dtype(v.dtype())?,
    };

    // Repeat q, k if num_v_heads > num_k_heads
    let repeat_factor = hv / hk;
    let (q, k) = if repeat_factor > 1 {
        // Repeat Q, K along head dim to match V's num_v_heads
        // tile repeats the ENTIRE tensor, not individual heads
        // We need repeat-interleave along axis 2 instead
        // For [B, T, Hk, Dk] → [B, T, Hv, Dk] where Hv = Hk * factor
        // Use reshape + tile + reshape
        let bq = q.dim(0);
        let tq = q.dim(1);
        let q_rep = q.reshape(&[bq, tq, hk as i32, 1, dk])?;
        let q_rep = mlx_rs::ops::tile(&q_rep, &[1, 1, 1, repeat_factor as i32, 1])?;
        let q_rep = q_rep.reshape(&[bq, tq, hv as i32, dk])?;
        let k_rep = k.reshape(&[bq, tq, hk as i32, 1, dk])?;
        let k_rep = mlx_rs::ops::tile(&k_rep, &[1, 1, 1, repeat_factor as i32, 1])?;
        let k_rep = k_rep.reshape(&[bq, tq, hv as i32, dk])?;
        (q_rep, k_rep)
    } else {
        (q.clone(), k.clone())
    };

    let mut outputs = Vec::with_capacity(t);

    for ti in 0..t {
        // Extract timestep ti: take along axis 1
        let idx = Array::from_int(ti as i32).reshape(&[1])?;
        let qt = mlx_rs::ops::squeeze_axes(
            &mlx_rs::ops::indexing::take_axis(&q, &idx, 1)?, &[1]
        )?;
        let kt = mlx_rs::ops::squeeze_axes(
            &mlx_rs::ops::indexing::take_axis(&k, &idx, 1)?, &[1]
        )?;
        let vt = mlx_rs::ops::squeeze_axes(
            &mlx_rs::ops::indexing::take_axis(v, &idx, 1)?, &[1]
        )?;
        let gt = mlx_rs::ops::squeeze_axes(
            &mlx_rs::ops::indexing::take_axis(g, &idx, 1)?, &[1]
        )?;
        let bt = mlx_rs::ops::squeeze_axes(
            &mlx_rs::ops::indexing::take_axis(beta, &idx, 1)?, &[1]
        )?;

        let old_state = state.clone();

        // Decay: state *= g[..., None, None]
        let g_exp = gt.reshape(&[b, hv as i32, 1, 1])?;
        state = &state * &g_exp;

        // kv_mem = sum_k(state * k[..., None, :]) → [B, Hv, Dv]
        let k_exp = kt.reshape(&[b, hv as i32, 1, dk])?;
        let kv_mem = mlx_rs::ops::sum_axis(&(&state * &k_exp), -1, Some(false))?;

        // delta = (v - kv_mem) * beta[..., None]
        let bt_exp = bt.reshape(&[b, hv as i32, 1])?;
        let delta = &(&vt - &kv_mem) * &bt_exp;

        // state += k[..., None, :] * delta[..., :, None]
        let delta_exp = mlx_rs::ops::expand_dims(&delta, -1)?;
        state = &state + &(&k_exp * &delta_exp);

        // out = sum_k(state * q[..., None, :]) → [B, Hv, Dv]
        let q_exp = qt.reshape(&[b, hv as i32, 1, dk])?;
        let out = mlx_rs::ops::sum_axis(&(&state * &q_exp), -1, Some(false))?;

        // Mask: revert state if masked
        if let Some(m) = mask {
            let mt_idx = Array::from_slice(&[ti as i32], &[1]);
            let mt = mlx_rs::ops::squeeze_axes(
                &mlx_rs::ops::indexing::take_axis(m, &mt_idx, 1)?, &[1]
            )?;
            let mt = mt.reshape(&[b, 1, 1, 1])?;
            state = mlx_rs::ops::r#where(&mt, &state, &old_state)?;
        }

        outputs.push(out);
    }

    let y = mlx_rs::ops::stack_axis(
        &outputs.iter().collect::<Vec<_>>(), 1
    )?;
    Ok((y, state))
}

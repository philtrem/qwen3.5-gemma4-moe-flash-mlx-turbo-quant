use mlx_rs::error::Exception;
use mlx_rs::Array;

use crate::cache::KVCache;
use crate::model::mlp::QuantizedLinear;
use crate::model::norm::RMSNorm;

/// Full attention layer (10 of 40 layers — every 4th).
#[allow(dead_code)]
pub struct Attention {
    pub q_proj: QuantizedLinear,
    pub k_proj: QuantizedLinear,
    pub v_proj: QuantizedLinear,
    pub o_proj: QuantizedLinear,
    pub q_norm: RMSNorm,
    pub k_norm: RMSNorm,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rope_dims: i32,
    pub rope_theta: f32,
    pub scale: f32,
}

impl Attention {
    pub fn forward(
        &self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut KVCache,
    ) -> Result<Array, Exception> {
        let b = x.dim(0);
        let l = x.dim(1);

        // Q: [B, L, num_heads * head_dim * 2] → split into queries + gate
        let q_out = self.q_proj.forward(x)?;
        let q_reshaped = q_out.reshape(&[b, l, self.num_heads as i32, -1])?;
        let parts = mlx_rs::ops::split(&q_reshaped, 2, Some(-1))?;
        let queries = &parts[0];
        let gate = parts[1].reshape(&[b, l, -1])?;

        // K, V projections
        let keys = self.k_proj.forward(x)?
            .reshape(&[b, l, self.num_kv_heads as i32, -1])?;
        let values = self.v_proj.forward(x)?
            .reshape(&[b, l, self.num_kv_heads as i32, -1])?;

        // Per-head RMS norm
        let queries = self.q_norm.forward(queries)?;
        let keys = self.k_norm.forward(&keys)?;

        // Transpose to [B, heads, L, head_dim]
        let queries = mlx_rs::ops::transpose_axes(&queries, &[0, 2, 1, 3])?;
        let keys = mlx_rs::ops::transpose_axes(&keys, &[0, 2, 1, 3])?;
        let values = mlx_rs::ops::transpose_axes(&values, &[0, 2, 1, 3])?;

        // Partial RoPE
        let offset = cache.offset() as i32;
        let queries = mlx_rs::fast::rope(
            &queries, self.rope_dims, false, Some(self.rope_theta),
            1.0, offset, None::<&Array>,
        )?;
        let keys = mlx_rs::fast::rope(
            &keys, self.rope_dims, false, Some(self.rope_theta),
            1.0, offset, None::<&Array>,
        )?;

        // KV cache update
        let (keys, values) = cache.update_and_fetch(keys, values)?;

        // SDPA
        let output = if let Some(m) = mask {
            mlx_rs::fast::scaled_dot_product_attention(
                &queries, &keys, &values, self.scale,
                mlx_rs::fast::ScaledDotProductAttentionMask::Array(m),
            )?
        } else {
            mlx_rs::fast::scaled_dot_product_attention(
                &queries, &keys, &values, self.scale,
                None::<mlx_rs::fast::ScaledDotProductAttentionMask>,
            )?
        };

        // Transpose back: [B, L, num_heads * head_dim]
        let output = mlx_rs::ops::transpose_axes(&output, &[0, 2, 1, 3])?;
        let output = output.reshape(&[b, l, -1])?;

        // Output gating: o_proj(output * sigmoid(gate))
        let gate = mlx_rs::ops::sigmoid(&gate)?;
        let gated = &output * &gate;
        self.o_proj.forward(&gated)
    }
}

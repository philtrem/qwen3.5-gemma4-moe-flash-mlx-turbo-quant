use mlx_rs::error::Exception;
use mlx_rs::Array;

use crate::cache::KVCache;
use crate::model::mlp::QuantizedLinear;
use crate::model::norm::{RMSNorm, RMSNormNoScale};

/// Gemma4 attention layer (both sliding and full attention).
///
/// Key differences from Qwen attention:
/// - No output gating (no sigmoid gate)
/// - K==V support for full attention layers (V = raw K before k_norm)
/// - V gets RMSNormNoScale (no learnable weight)
/// - scale = 1.0 (not 1/sqrt(d))
/// - Different head_dim and num_kv_heads per layer type
pub struct Gemma4Attention {
    pub q_proj: QuantizedLinear,
    pub k_proj: QuantizedLinear,
    pub v_proj: Option<QuantizedLinear>, // None when use_k_eq_v
    pub o_proj: QuantizedLinear,
    pub q_norm: RMSNorm,
    pub k_norm: RMSNorm,
    pub v_norm: RMSNormNoScale,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rope_dims: i32,
    pub rope_theta: f32,
    pub use_k_eq_v: bool,
}

impl Gemma4Attention {
    pub fn forward(
        &self,
        x: &Array,
        mask: Option<&Array>,
        cache: &mut KVCache,
    ) -> Result<Array, Exception> {
        let b = x.dim(0);
        let l = x.dim(1);

        // Q projection + reshape + norm
        let queries = self.q_proj.forward(x)?
            .reshape(&[b, l, self.num_heads as i32, self.head_dim as i32])?;
        let queries = self.q_norm.forward(&queries)?;

        // K projection + reshape
        let keys = self.k_proj.forward(x)?
            .reshape(&[b, l, self.num_kv_heads as i32, self.head_dim as i32])?;

        // V: either from v_proj or K==V (raw K before k_norm)
        let values = if self.use_k_eq_v {
            keys.clone()
        } else {
            self.v_proj.as_ref().unwrap().forward(x)?
                .reshape(&[b, l, self.num_kv_heads as i32, self.head_dim as i32])?
        };

        // K norm (after V capture for K==V)
        let keys = self.k_norm.forward(&keys)?;
        // V norm (no learnable scale)
        let values = self.v_norm.forward(&values)?;

        // Transpose to [B, heads, L, head_dim]
        let keys = mlx_rs::ops::transpose_axes(&keys, &[0, 2, 1, 3])?;
        let values = mlx_rs::ops::transpose_axes(&values, &[0, 2, 1, 3])?;
        let queries = mlx_rs::ops::transpose_axes(&queries, &[0, 2, 1, 3])?;

        // RoPE
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

        // SDPA with scale=1.0
        let output = if let Some(m) = mask {
            mlx_rs::fast::scaled_dot_product_attention(
                &queries, &keys, &values, 1.0,
                mlx_rs::fast::ScaledDotProductAttentionMask::Array(m),
            )?
        } else {
            mlx_rs::fast::scaled_dot_product_attention(
                &queries, &keys, &values, 1.0,
                None::<mlx_rs::fast::ScaledDotProductAttentionMask>,
            )?
        };

        // Transpose back: [B, L, num_heads * head_dim]
        let output = mlx_rs::ops::transpose_axes(&output, &[0, 2, 1, 3])?;
        let output = output.reshape(&[b, l, -1])?;

        // Output projection (no gating unlike Qwen)
        self.o_proj.forward(&output)
    }

    /// Speculative attention: runs Q/K/V projections, concatenates with existing
    /// cached K/V (virtual append), runs SDPA, but does NOT modify the cache.
    /// Returns None if cached_kv is None (empty cache during prefill).
    pub fn forward_speculative(
        &self,
        x: &Array,
        mask: Option<&Array>,
        cached_kv: Option<(Array, Array)>,
        offset: usize,
    ) -> Result<Array, Exception> {
        let b = x.dim(0);
        let l = x.dim(1);

        let queries = self.q_proj.forward(x)?
            .reshape(&[b, l, self.num_heads as i32, self.head_dim as i32])?;
        let queries = self.q_norm.forward(&queries)?;

        let keys = self.k_proj.forward(x)?
            .reshape(&[b, l, self.num_kv_heads as i32, self.head_dim as i32])?;
        let values = if self.use_k_eq_v {
            keys.clone()
        } else {
            self.v_proj.as_ref().unwrap().forward(x)?
                .reshape(&[b, l, self.num_kv_heads as i32, self.head_dim as i32])?
        };
        let keys = self.k_norm.forward(&keys)?;
        let values = self.v_norm.forward(&values)?;

        let keys = mlx_rs::ops::transpose_axes(&keys, &[0, 2, 1, 3])?;
        let values = mlx_rs::ops::transpose_axes(&values, &[0, 2, 1, 3])?;
        let queries = mlx_rs::ops::transpose_axes(&queries, &[0, 2, 1, 3])?;

        // RoPE at the speculative position (same offset as real cache)
        let queries = mlx_rs::fast::rope(
            &queries, self.rope_dims, false, Some(self.rope_theta),
            1.0, offset as i32, None::<&Array>,
        )?;
        let keys = mlx_rs::fast::rope(
            &keys, self.rope_dims, false, Some(self.rope_theta),
            1.0, offset as i32, None::<&Array>,
        )?;

        // Virtual append: concatenate with existing cache (no mutation)
        let (keys, values) = if let Some((ck, cv)) = cached_kv {
            let keys = mlx_rs::ops::concatenate_axis(&[&ck, &keys], 2)?;
            let values = mlx_rs::ops::concatenate_axis(&[&cv, &values], 2)?;
            (keys, values)
        } else {
            (keys, values)
        };

        // SDPA with scale=1.0
        let output = if let Some(m) = mask {
            mlx_rs::fast::scaled_dot_product_attention(
                &queries, &keys, &values, 1.0,
                mlx_rs::fast::ScaledDotProductAttentionMask::Array(m),
            )?
        } else {
            mlx_rs::fast::scaled_dot_product_attention(
                &queries, &keys, &values, 1.0,
                None::<mlx_rs::fast::ScaledDotProductAttentionMask>,
            )?
        };

        let output = mlx_rs::ops::transpose_axes(&output, &[0, 2, 1, 3])?;
        let output = output.reshape(&[b, l, -1])?;
        self.o_proj.forward(&output)
    }
}

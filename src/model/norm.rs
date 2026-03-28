use mlx_rs::error::Exception;
use mlx_rs::Array;

/// RMSNorm — wraps mlx_rs::fast::rms_norm.
pub struct RMSNorm {
    pub weight: Array,
    pub eps: f32,
}

impl RMSNorm {
    pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
        mlx_rs::fast::rms_norm(x, &self.weight, self.eps)
    }
}

/// RMSNormGated — rms_norm with silu gating (precise swiglu in float32).
pub struct RMSNormGated {
    pub weight: Array,
    pub eps: f32,
}

impl RMSNormGated {
    /// out = silu(gate.float32()) * rms_norm(x).float32(), cast back to input dtype.
    pub fn forward(&self, x: &Array, gate: &Array) -> Result<Array, Exception> {
        let normed = mlx_rs::fast::rms_norm(x, &self.weight, self.eps)?;
        let orig_dtype = x.dtype();
        let gate_f32 = gate.as_dtype(mlx_rs::Dtype::Float32)?;
        let gate_act = mlx_rs::nn::silu(&gate_f32)?;
        let normed_f32 = normed.as_dtype(mlx_rs::Dtype::Float32)?;
        let result = &gate_act * &normed_f32;
        result.as_dtype(orig_dtype)
    }
}

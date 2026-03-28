use mlx_rs::error::Exception;
use mlx_rs::Array;

/// Quantized linear layer — 8-bit weights with scales and biases.
pub struct QuantizedLinear {
    pub weight: Array,  // uint32 packed
    pub scales: Array,  // bfloat16
    pub biases: Array,  // bfloat16
    pub bits: i32,
    pub group_size: i32,
}

impl QuantizedLinear {
    pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
        mlx_rs::ops::quantized_matmul(
            x,
            &self.weight,
            &self.scales,
            &self.biases,
            Some(true),
            Some(self.group_size),
            Some(self.bits),
        )
    }
}

/// Standard MLP: SiLU(gate_proj(x)) * up_proj(x) → down_proj
pub struct MLP {
    pub gate_proj: QuantizedLinear,
    pub up_proj: QuantizedLinear,
    pub down_proj: QuantizedLinear,
}

impl MLP {
    pub fn forward(&self, x: &Array) -> Result<Array, Exception> {
        let gate = mlx_rs::nn::silu(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        let h = &gate * &up;
        self.down_proj.forward(&h)
    }
}

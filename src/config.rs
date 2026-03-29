use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct QuantizationConfig {
    pub bits: u32,
    pub group_size: u32,
    #[serde(default)]
    pub mode: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct TextModelArgs {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default)]
    pub intermediate_size: Option<usize>,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_full_attn_interval")]
    pub full_attention_interval: usize,

    // Linear attention (GatedDeltaNet)
    #[serde(default = "default_32")]
    pub linear_num_value_heads: usize,
    #[serde(default = "default_16")]
    pub linear_num_key_heads: usize,
    #[serde(default = "default_128")]
    pub linear_key_head_dim: usize,
    #[serde(default = "default_128")]
    pub linear_value_head_dim: usize,
    #[serde(default = "default_4")]
    pub linear_conv_kernel_dim: usize,

    // RoPE
    #[serde(default = "default_partial_rotary")]
    pub partial_rotary_factor: f64,
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,

    // EOS tokens
    #[serde(default)]
    pub eos_token_id: EosTokenId,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
#[allow(dead_code)]
pub enum EosTokenId {
    Single(u32),
    Multiple(Vec<u32>),
}

impl Default for EosTokenId {
    fn default() -> Self {
        EosTokenId::Multiple(vec![248046, 248044])
    }
}

impl EosTokenId {
    #[allow(dead_code)]
    pub fn ids(&self) -> Vec<u32> {
        match self {
            EosTokenId::Single(id) => vec![*id],
            EosTokenId::Multiple(ids) => ids.clone(),
        }
    }
}

fn default_rope_theta() -> f64 {
    10_000_000.0
}
fn default_partial_rotary() -> f64 {
    0.25
}
fn default_true() -> bool {
    true
}
fn default_full_attn_interval() -> usize {
    4
}
fn default_32() -> usize {
    32
}
fn default_16() -> usize {
    16
}
fn default_128() -> usize {
    128
}
fn default_4() -> usize {
    4
}

impl TextModelArgs {
    pub fn from_config_file(path: &Path) -> anyhow::Result<(Self, Option<QuantizationConfig>)> {
        let content = std::fs::read_to_string(path)?;
        let config: serde_json::Value = serde_json::from_str(&content)?;

        let text_config = config.get("text_config").unwrap_or(&config);
        let args: TextModelArgs = serde_json::from_value(text_config.clone())?;

        let quant = config
            .get("quantization")
            .map(|q| serde_json::from_value(q.clone()))
            .transpose()?;

        Ok((args, quant))
    }

    pub fn is_linear_layer(&self, layer_idx: usize) -> bool {
        (layer_idx + 1) % self.full_attention_interval != 0
    }

    pub fn rope_dims(&self) -> i32 {
        (self.head_dim as f64 * self.partial_rotary_factor) as i32
    }

    /// Key dimension for linear attention: num_k_heads * key_head_dim
    pub fn key_dim(&self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim
    }

    /// Value dimension for linear attention: num_v_heads * value_head_dim
    pub fn value_dim(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    /// Conv dimension for linear attention: key_dim*2 + value_dim
    pub fn conv_dim(&self) -> usize {
        self.key_dim() * 2 + self.value_dim()
    }
}

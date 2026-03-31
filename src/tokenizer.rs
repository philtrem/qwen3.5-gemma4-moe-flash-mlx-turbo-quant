use std::path::Path;

use anyhow::Result;

pub struct QwenTokenizer {
    tokenizer: tokenizers::Tokenizer,
    chat_env: minijinja::Environment<'static>,
    pub eos_token_ids: Vec<u32>,
}

impl QwenTokenizer {
    pub fn from_dir(path: &Path) -> Result<Self> {
        let tokenizer =
            tokenizers::Tokenizer::from_file(path.join("tokenizer.json"))
                .map_err(|e| anyhow::anyhow!("tokenizer load error: {}", e))?;

        let mut env = minijinja::Environment::new();
        let template_path = path.join("chat_template.jinja");
        if template_path.exists() {
            let template = std::fs::read_to_string(&template_path)?;
            env.add_template_owned("chat".to_string(), template)?;
        }

        // Read EOS token IDs from config.json
        let config_path = path.join("config.json");
        let eos_token_ids = if config_path.exists() {
            let config: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;
            match config.get("eos_token_id") {
                Some(serde_json::Value::Array(ids)) => ids
                    .iter()
                    .filter_map(|v| v.as_u64().map(|n| n as u32))
                    .collect(),
                Some(serde_json::Value::Number(n)) => {
                    vec![n.as_u64().unwrap_or(248044) as u32]
                }
                _ => vec![248046, 248044],
            }
        } else {
            vec![248046, 248044]
        };

        Ok(Self {
            tokenizer,
            chat_env: env,
            eos_token_ids,
        })
    }

    pub fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let enc = self.tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenizer encode failed: {}", e))?;
        Ok(enc.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32]) -> anyhow::Result<String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| anyhow::anyhow!("tokenizer decode failed: {}", e))
    }

    pub fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<String> {
        if self.chat_env.get_template("chat").is_err() {
            // Fallback: simple ChatML format
            let mut parts = Vec::new();
            for msg in messages {
                parts.push(format!(
                    "<|im_start|>{}\n{}<|im_end|>",
                    msg.role, msg.content
                ));
            }
            parts.push("<|im_start|>assistant\n".to_string());
            return Ok(parts.join("\n"));
        }

        let template = self.chat_env.get_template("chat")?;
        let rendered = template.render(minijinja::context! {
            messages => messages,
            add_generation_prompt => true,
        })?;
        Ok(rendered)
    }

    pub fn is_eos(&self, token_id: u32) -> bool {
        self.eos_token_ids.contains(&token_id)
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

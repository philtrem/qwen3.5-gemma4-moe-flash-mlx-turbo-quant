mod cache;
mod config;
mod engine;
mod ffi;
mod memory;
mod model;
mod perf;
mod splitter;
mod tokenizer;

use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "flash-qwen")]
#[command(about = "All-Rust UMA-native MoE inference for Qwen3.5-35B-A3B")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Split original model into resident weights + per-layer expert files
    Split {
        #[arg(long)]
        model_path: PathBuf,
        #[arg(long)]
        output_path: PathBuf,
        /// Expert file format (only "ecb" is supported)
        #[arg(long, default_value = "ecb")]
        format: String,
    },
    /// Generate text from a prompt
    Generate {
        #[arg(long)]
        model_path: PathBuf,
        #[arg(long)]
        tokenizer_path: PathBuf,
        #[arg(long, default_value = "Hello")]
        prompt: String,
        #[arg(long, default_value_t = 256)]
        max_tokens: usize,
        #[arg(long, default_value_t = 0.7)]
        temperature: f32,
        #[arg(long, default_value_t = 0.9)]
        top_p: f32,
        #[arg(long)]
        warm_experts: Option<PathBuf>,
        /// TurboQuant KV cache quantization bits (2, 3, or 4). Omit for bf16.
        #[arg(long)]
        kv_quant_bits: Option<u8>,
        /// Disable speculative F_RDADVISE prefetch for next-layer predicted experts.
        #[arg(long)]
        no_speculate: bool,
        /// Load warm set at startup (preads frequent experts into page cache).
        #[arg(long)]
        warm_set: bool,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Split {
            model_path,
            output_path,
            format,
        } => {
            eprintln!("Splitting model: {} → {} (format: {})", model_path.display(), output_path.display(), format);
            splitter::split_model(&model_path, &output_path, &format)?;
            eprintln!("Done.");
        }

        Command::Generate {
            model_path,
            tokenizer_path,
            prompt,
            max_tokens,
            temperature,
            top_p,
            warm_experts,
            kv_quant_bits,
            no_speculate,
            warm_set,
        } => {
            // Load config
            let config_path = model_path.join("config.json");
            let (args, quant) = config::TextModelArgs::from_config_file(&config_path)?;

            // Load tokenizer
            eprintln!("Loading tokenizer from {}...", tokenizer_path.display());
            let tokenizer = tokenizer::QwenTokenizer::from_dir(&tokenizer_path)?;

            // Apply chat template
            let chat_prompt = tokenizer
                .apply_chat_template(&[tokenizer::ChatMessage {
                    role: "user".to_string(),
                    content: prompt.clone(),
                }])
                .unwrap_or_else(|_| {
                    format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", prompt)
                });

            // Create ExpertMemoryManager — mmaps expert files, used for on-demand loading
            let expert_dir = model_path.join("experts");
            eprintln!("Mapping expert files from {}...", expert_dir.display());
            let mut mem_mgr = memory::ExpertMemoryManager::new(&expert_dir, args.num_hidden_layers)?;

            // Load model FIRST (resident weights only — no expert arrays).
            // This allocates 2.76 GB of Metal buffers. Loading before warm set
            // prefetch ensures madvise pages aren't evicted by weight allocation.
            eprintln!("Loading model from {}...", model_path.display());
            let mut model = model::load_model(&model_path, &args, quant.as_ref())?;

            // Prefetch warm set into page cache AFTER model loading
            let warm_path = warm_experts
                .or_else(|| {
                    let auto = model_path.join("warm_experts.json");
                    if auto.exists() { Some(auto) } else { None }
                });

            if let Some(wp) = warm_path.filter(|_| warm_set) {
                eprintln!("Prefetching warm set from {}...", wp.display());
                let warm: serde_json::Value =
                    serde_json::from_str(&std::fs::read_to_string(&wp)?)?;
                let experts: Vec<(u32, u32)> = warm["experts"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|pair| {
                        let arr = pair.as_array().unwrap();
                        (arr[0].as_u64().unwrap() as u32, arr[1].as_u64().unwrap() as u32)
                    })
                    .collect();

                let advised = mem_mgr.mlock_warm_set(&experts);
                mem_mgr.set_warm_set(&experts);
                eprintln!(
                    "Warm set: {} experts, prefetched {:.1} GB",
                    experts.len(),
                    advised as f64 / 1e9
                );
            }

            // Generate
            if let Some(bits) = kv_quant_bits {
                eprintln!("TurboQuant KV cache: {}-bit", bits);
            }
            let speculate = !no_speculate;
            eprintln!("Engine ready.\n");
            if no_speculate {
                eprintln!("Speculative prediction: disabled");
            }
            let _output = engine::generate(
                &mut model,
                &tokenizer,
                &chat_prompt,
                max_tokens,
                temperature,
                top_p,
                &mem_mgr,
                kv_quant_bits,
                speculate,
            )?;
        }
    }

    Ok(())
}

# AGENTS.md

## Git

Do NOT add Codex as a co-author in commit messages.

## Context

Detailed implementation history, decisions, and performance data are in the auto-memory system. Check MEMORY.md before exploring the codebase — it likely has what you need.

## What this is

Flash-loading inference engine for Qwen3.5-35B-A3B (36.3 GB, 9-bit MLX) on Mac M4 base 16GB. All-Rust single binary via `mlx-rs`, on-demand expert loading from mmap.

## Build

```bash
cargo build --release
```

## Run

```bash
# Split model (one-time, creates safetensors format)
./target/release/flash-qwen split \
  --model-path /Users/philtrem/.lmstudio/models/inferencerlabs/Qwen3.5-35B-A3B-MLX-9bit \
  --output-path ./split_model_st

# Generate (warm set auto-detected from split_model_st/warm_experts.json)
./target/release/flash-qwen generate \
  --model-path ./split_model_st \
  --tokenizer-path /Users/philtrem/.lmstudio/models/inferencerlabs/Qwen3.5-35B-A3B-MLX-9bit \
  --prompt "Hello" --max-tokens 256
```

## Architecture

### `src/`
- **main.rs** — CLI (clap): split + generate subcommands
- **model/** — Model/TextModel/DecoderLayer, GatedDeltaNet, Attention, SparseMoeBlock, RMSNorm, MLP
- **memory.rs** — ExpertMemoryManager: mmap expert safetensors, on-demand extraction per-expert, warm set madvise
- **engine.rs** — generate() loop + nucleus sampling
- **ffi.rs** — gather_qmm FFI wrapper via mlx-sys
- **splitter.rs** — model splitter (original → resident + per-layer expert safetensors)
- Expert weights loaded on-demand: mmap → extract active experts (~27 MB/layer) → Array::from_raw_data → gather_qmm
- Per-layer eval barriers at CPU/GPU boundaries (argsort is CPU, gather_qmm is GPU)

### Model
- Model type is `qwen3_5_moe` mapping to `mlx_lm.models.qwen3_5` (NOT `qwen3_next`)
- 40 layers: 30 linear-attention (GatedDeltaNet/ArraysCache) + 10 full-attention (Attention/KVCache), every 4th layer is full-attention
- Weights are already sanitized (mlx-sanitized: 0.30.7) — do NOT re-sanitize
- Expert dimensions: hidden=2048, intermediate=512, 256 experts/layer, top_k=8, 8-bit quant, group_size=32

## Performance

- 1.3 tok/s decode, 0.6 tok/s prefill. No swap storms. Peak expert memory ~27 MB per layer.

## Key gotchas

### MLX / mlx-rs
- `mx.linalg.qr` requires `stream=mx.cpu` — not supported on GPU
- MLX has no `searchsorted` — use boundary comparison: `(x[..., None] > boundaries).sum(-1)`
- `mlx_array_new_data` (and `Array::from_raw_data`) **copies** data — no zero-copy from mmap
- `Array::load_safetensors()` creates lazy arrays; when evaluated, reads file data into anonymous Metal buffers (NOT mmap-backed). Loading all 40 expert files (34.6 GB) causes swap storms on 16 GB.
- `gather_qmm` is NOT in mlx-rs — use `mlx_sys::mlx_gather_qmm` via FFI wrapper
- `argsort` runs on CPU; eval boundaries needed before GPU gather_qmm
- Activation dtype drift: bf16×f32 scalar promotes to f32 — cast scalars to input dtype

### Memory / UMA
- **Do NOT load all expert files via load_safetensors** — causes 25+ GB swap on 16 GB
- On-demand expert extraction from mmap is the correct approach (~27 MB per layer, not 864 MB)
- madvise(MADV_WILLNEED) for warm set prefetch, NOT mlock (mlock of 10+ GB causes swap)
- Per-layer eval ensures expert arrays are freed after each layer (peak ~27 MB, not cumulative)


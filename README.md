# flash-moe

All-Rust inference engine for sparse Mixture-of-Experts models on Apple Silicon. Runs large MoE models on an M4 Mac Mini with 16 GB of RAM by loading experts on-demand from SSD.

**Supported models:**
- **Gemma 4 26B-A4B** — 26B params, 4B active. 30 layers, 128 experts, top-8.
- **Qwen 3.5 35B-A3B** — 35B params, 3B active. 40 layers, 256 experts, top-8.

The key idea: sparse MoE models only activate a small fraction of parameters per token (8 experts out of 128–256 per layer), so the full model doesn't need to fit in memory. flash-moe keeps resident weights in Metal buffers and loads experts on-demand from SSD via memory-mapped I/O, using GCD-dispatched prefetch to keep pages ahead of the GPU.

## How it works

The model is split into two parts:

- **Resident weights** (~2.8–3.6 GB): embeddings, attention, norms, router, dense MLP, output head. Loaded once into Metal buffers at startup.
- **Expert files** (~428–453 MB/layer): one ECB (expert-centric binary) file per layer. Memory-mapped but never fully loaded — only the 8 active experts are paged in per layer per token.

### I/O pipeline

The bottleneck isn't compute — it's getting expert bytes from SSD to GPU before it stalls. Without explicit prefetch, the GPU triggers page faults that pull data in 16 KB chunks — synchronous kernel traps that reduce effective throughput to a fraction of what sequential reads achieve. flash-moe avoids this with a two-stage GCD prefetch pipeline:

1. **Speculative** (during GPU eval): After submitting the current layer to the GPU, fire off low-priority (utility QoS) GCD workers to prefault pages for the *next* layer's predicted experts.
2. **Reactive** (after routing): Once the router picks the actual 8 experts, cancel any in-flight speculative work (atomic flag — no SSD contention), then dispatch high-priority (userInitiated QoS) workers to prefault the exact pages needed. Blocks until all pages are resident.
3. **Eval** (zero faults): GPU reads from Metal buffers backed by already-resident mmap pages. Pure compute, no page faults.

Cancellation is what makes this work — without it, speculative I/O contends with reactive and throughput drops significantly.

### Per-token I/O

| Model | Expert size | Active/layer | Layers | I/O per token |
|-------|------------|-------------|--------|--------------|
| Gemma 4 26B | 3.35 MB | 26.8 MB | 30 | 803 MB |
| Qwen 3.5 35B | 1.77 MB | 14.2 MB | 40 | 566 MB |

Gemma 4 26B-A4B is actually heavier on I/O per token than Qwen 3.5 35B-A3B despite being a smaller model. Each expert is ~2× bigger (3.35 MB vs 1.77 MB) because of wider hidden dimensions (2816×704 vs 2048×512). With 30 MoE layers × 8 active experts, that's 803 MB read from SSD per token — 42% more than Qwen's 566 MB. On a 16 GB M4 where I/O is the bottleneck, that's the whole story.

### Why not just load everything into RAM?

These models are 13–19 GB at 4-bit. On a 16 GB machine, that means swap, and swap means page faults during GPU eval — which is exactly what this project avoids. The MoE sparsity (3–6% active) makes on-demand loading viable: you only need the data you're actually using.

## Requirements

- **macOS** on Apple Silicon (tested on M4 Mac Mini, 16 GB)
- **Rust** toolchain (stable)
- **Model weights** (one of):
  - [philtrem/gemma-4-26b-a4b-it-MLX-4bit](https://huggingface.co/philtrem/gemma-4-26b-a4b-it-MLX-4bit) (~13 GB)
  - [mlx-community/Qwen3.5-35B-A3B-4bit](https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-4bit) (~19 GB)

## Build

```bash
cargo build --release
```

## Usage

### 1. Download the model

```bash
# Gemma 4 (default)
huggingface-cli download philtrem/gemma-4-26b-a4b-it-MLX-4bit \
  --local-dir ./gemma-4-26b-a4b-it-MLX-4bit

# Or Qwen 3.5
huggingface-cli download mlx-community/Qwen3.5-35B-A3B-4bit \
  --local-dir ./Qwen3.5-35B-A3B-4bit
```

### 2. Split the model

Converts HuggingFace safetensors into resident weights + per-layer expert ECB files:

```bash
# Gemma 4
./target/release/flash-moe split \
  --model-path ./gemma-4-26b-a4b-it-MLX-4bit \
  --output-path ./split_gemma4

# Qwen 3.5
./target/release/flash-moe split \
  --model-path ./Qwen3.5-35B-A3B-4bit \
  --output-path ./split_qwen
```

One-time step. You can delete the original download after splitting.

### 3. Generate

```bash
# Gemma 4 (default — model-path defaults to ./split_gemma4)
./target/release/flash-moe generate \
  --prompt "Explain the Riemann hypothesis in simple terms" \
  --max-tokens 256

# Qwen 3.5
./target/release/flash-moe generate \
  --model-path ./split_qwen \
  --tokenizer-path ./Qwen3.5-35B-A3B-4bit \
  --prompt "Hello" --max-tokens 256
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--temperature` | 0.7 | Sampling temperature |
| `--top-p` | 0.9 | Nucleus sampling threshold |
| `--no-speculate` | off | Disable speculative prefetch for predicted experts |
| `--warm-set` | off | Pread frequent experts into page cache at startup |
| `--kv-quant-bits N` | 3 | TurboQuant KV cache: 2, 3, or 4-bit quantization |
| `--no-kv-quant` | off | Disable KV cache quantization (plain bf16) |

## License

MIT

# flash-qwen

All-Rust inference engine for [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B) on Apple Silicon. Runs a 35B-parameter Mixture-of-Experts model on an M4 Mac Mini with 16 GB of RAM.

The key idea: Qwen3.5-35B-A3B only activates 3B parameters per token (8 of 256 experts per layer), so the full model doesn't need to fit in memory. flash-qwen keeps resident weights in Metal buffers and loads experts on-demand from SSD via memory-mapped I/O, using GCD-dispatched prefetch to keep pages ahead of the GPU.

While built around Qwen3.5, the engine is designed as a foundation for serving large MoE models on M-series Macs — the SSD prefetch pipeline, zero-copy Metal path, and on-demand expert loading are model-agnostic.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        CLI (clap)                       │
│                   split / generate                      │
├─────────────────────────────────────────────────────────┤
│  engine.rs         Token loop + nucleus sampling        │
├─────────────────────────────────────────────────────────┤
│  model/            Qwen3.5 MoE (mlx-rs)                 │
│    mod.rs          TextModel, DecoderLayer              │
│    attention.rs    Full attention (every 4th layer)     │
│    gated_delta.rs  GatedDeltaNet linear attention       │
│    moe.rs          SparseMoeBlock (256 experts, top-8)  │
│    mlp.rs          Expert MLP (gate/up/down proj.)      │
│    norm.rs         RMSNorm                              │
├─────────────────────────────────────────────────────────┤
│  memory.rs         ExpertMemoryManager                  │
│                    ├ GCD speculative prefetch (utility) │
│                    ├ GCD reactive prefetch (userInit)   │
│                    ├ Zero-copy mmap → Metal buffers     │
│                    └ Warm set pread (optional)          │
├─────────────────────────────────────────────────────────┤
│  cache.rs          KV cache + TurboQuant (optional)     │
│  ffi.rs            gather_qmm FFI + array_from_mmap     │
│  ffi_zerocopy.cpp  MLX Metal newBufferWithBytesNoCopy   │
│  splitter.rs       Model → resident + per-layer ECB     │
│  perf.rs           Per-phase timing (routing, eval, I/O)│
└─────────────────────────────────────────────────────────┘
         ▼ SSD (mmap)                    ▲ Metal GPU
   ┌──────────────┐              ┌──────────────┐
   │  Expert ECB  │  ──pages──▶  │  GPU eval    │
   │  files/layer │              │ (fault-free) │
   └──────────────┘              └──────────────┘
```

**40 layers**: 30 linear-attention (GatedDeltaNet) + 10 full-attention, every 4th layer. Each MoE layer has 256 experts (hidden=2048, intermediate=512); the router picks 8 per token. At 4-bit quantization, that's ~13.5 MB of expert data per layer, loaded from SSD via prefaulted mmap pages and handed to the GPU as zero-copy Metal buffers.

### I/O pipeline

The bottleneck isn't compute — it's getting expert bytes from SSD to GPU before it stalls. Without explicit prefetch, the GPU triggers page faults that pull data in 16 KB chunks — synchronous kernel traps that reduce effective SSD throughput to a fraction of what sequential reads achieve. flash-qwen avoids this with a two-stage GCD prefetch pipeline:

1. **Speculative** (during GPU eval): After submitting the current layer to the GPU, fire off low-priority (utility QoS) GCD workers to prefault pages for the *next* layer's predicted experts. Uses routing pre-MoE signals for ~85% accuracy.
2. **Reactive** (after routing): Once the router picks the actual 8 experts, cancel any in-flight speculative work (generation counter — no SSD contention), then dispatch high-priority (userInitiated QoS) workers to prefault the exact pages needed. Blocks until all pages are resident.
3. **Eval** (zero faults): GPU reads from Metal buffers backed by already-resident mmap pages. Pure compute, no page faults.

Cancellation is what makes this work — without it, speculative I/O contends with reactive and throughput drops significantly.

## Requirements

- **macOS** on Apple Silicon (tested on M4 Mac Mini, 16 GB)
- **Rust** toolchain (stable)
- **Model weights**: [mlx-community/Qwen3.5-35B-A3B-4bit](https://huggingface.co/mlx-community/Qwen3.5-35B-A3B-4bit) (~19 GB after splitting)
- C++ compiler (for the Metal zero-copy shim, built automatically via `build.rs`)

## Build

```bash
cargo build --release
```

## Usage

### 1. Split the model

Converts HuggingFace safetensors into resident weights + per-layer expert ECB files for on-demand loading:

```bash
./target/release/flash-qwen split \
  --model-path /path/to/Qwen3.5-35B-A3B-4bit \
  --output-path ./split_model
```

This is a one-time step. The split output is ~19 GB for the 4-bit model.

### 2. Generate

```bash
./target/release/flash-qwen generate \
  --model-path ./split_model \
  --tokenizer-path /path/to/Qwen3.5-35B-A3B-4bit \
  --prompt "Explain the Riemann hypothesis in simple terms" \
  --max-tokens 256
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--temperature` | 0.7 | Sampling temperature |
| `--top-p` | 0.9 | Nucleus sampling threshold |
| `--no-speculate` | off | Disable speculative prefetch for predicted experts |
| `--warm-set` | off | Pread frequent experts into page cache at startup |
| `--kv-quant-bits N` | off (bf16) | TurboQuant KV cache: 2, 3, or 4-bit quantization |

## How it works

The model is split into two parts:

- **Resident weights** (~2.8 GB): embeddings, attention, norms, router weights, output head. Loaded once into Metal buffers at startup.
- **Expert files** (~450 MB/layer, 30 MoE layers): one ECB (expert-centric binary) file per layer containing all 256 experts in a contiguous layout. Memory-mapped but never fully loaded — only the 8 active experts (~13.5 MB) are paged in per layer per token.

On a 16 GB machine, the resident weights plus OS overhead leave roughly 10-12 GB for page cache. Since each token only touches ~400 MB of expert data across all layers (30 layers x 13.5 MB), and the SSD delivers ~3 GB/s, the working set fits comfortably in the page cache after the first few tokens. The GCD prefetch pipeline ensures pages are resident before the GPU needs them, eliminating page fault stalls.

### Why not just load everything into RAM?

35B parameters at 4-bit is ~19 GB. On a 16 GB machine, that means swap, and swap means page faults during GPU eval — which is exactly what this project avoids. The MoE sparsity (8/256 = 3.1% active) makes on-demand loading viable: you only need the data you're actually using.

### Why not use the Neural Engine?

The M4's ANE isn't useful here. I/O is the bottleneck — even instant compute wouldn't dramatically change throughput. But beyond that, ANE doesn't support 4-bit quantized matmul (it handles float16/int8 via CoreML), so you'd have to dequantize to float16 first, doubling memory traffic. ANE dispatch latency is also tuned for large-batch CoreML inference, not single-token autoregressive decode where GPU compute is already a minority of wall time.

## License

MIT

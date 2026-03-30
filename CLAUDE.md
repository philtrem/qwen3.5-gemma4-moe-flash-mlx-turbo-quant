# CLAUDE.md

## Git

Do NOT add Claude as a co-author in commit messages.

## Context

Detailed implementation history, decisions, and performance data are in the auto-memory system. Check MEMORY.md before exploring the codebase — it likely has what you need.

## What this is

Flash-loading inference engine for Qwen3.5-35B-A3B on Mac M4 base 16GB. All-Rust single binary via `mlx-rs`, on-demand expert loading via pread() or zero-copy mmap Metal buffers.

## Build

```bash
cargo build --release
```

## Run

```bash
# Split model (one-time, creates ECB format)
./target/release/flash-qwen split \
  --model-path /Users/philtrem/.lmstudio/models/mlx-community/Qwen3.5-35B-A3B-4bit \
  --output-path ./split_model_st

# Generate (best config is the default — no flags needed)
./target/release/flash-qwen generate \
  --model-path ./split_model_st \
  --tokenizer-path /Users/philtrem/.lmstudio/models/mlx-community/Qwen3.5-35B-A3B-4bit \
  --prompt "Hello" --max-tokens 256

# Negative flags to disable features:
#   --no-speculate    Disable F_RDADVISE speculative prefetch
#   --no-pipeline     Disable async_eval (use sync eval)
#   --no-warm-set     Skip warm set loading at startup
#   --kv-quant-bits 3 TurboQuant KV cache (3-bit)
```

## Architecture

### `src/`
- **main.rs** — CLI (clap): split + generate subcommands. Reads quant config from config.json. All flags are negative (`--no-*`).
- **model/** — Model/TextModel/DecoderLayer, GatedDeltaNet, Attention, SparseMoeBlock, RMSNorm, MLP
- **memory.rs** — ExpertMemoryManager: pread + zero-copy extraction, warm set pread prefetch, F_RDADVISE, double-buffered hybrid pread
- **engine.rs** — generate() loop + nucleus sampling
- **perf.rs** — PerfStats: per-phase timing accumulator (routing eval, layer eval, GPU wait, extract, routing CPU)
- **ffi.rs** — gather_qmm FFI + `array_from_mmap` zero-copy wrapper
- **ffi_zerocopy.cpp** — C++ shim: MLX array from mmap via Metal `newBufferWithBytesNoCopy`
- **splitter.rs** — model splitter (original → resident + per-layer expert ECB)
- **build.rs** — compiles ffi_zerocopy.cpp with MLX C++ headers

### I/O strategy (current default, USE_ZEROCOPY=true)
- **Reactive blocking pread** (best config): After routing eval determines actual experts, `pread_experts_sync()` does parallel pread (global rayon pool, 12 threads) into throwaway buffers to warm page cache. Blocks until all pages resident. Then mmap zerocopy arrays are created — GPU eval is fault-free. Gated by `NOREACTIVE=1` env var for A/B testing.
- **Speculative F_RDADVISE** (on by default, `--no-speculate` to disable): After reactive pread, `async_eval(h)` submits GPU. Then F_RDADVISE hints kernel to start reading L+1's predicted experts during GPU eval. **Net negative on 4-bit** in all tested variants — see exhaustive benchmark below.
- **Warm set pread at startup**: `mlock_warm_set()` uses parallel pread (not madvise) to guarantee warm expert pages are resident. Saves ~20ms/tok on reactive pread (cache hits vs SSD).
- Per-layer eval via `async_eval` + `eval` separates GPU submission from wait time in perf stats.

### Model
- Model type is `qwen3_5_moe` mapping to `mlx_lm.models.qwen3_5` (NOT `qwen3_next`)
- 40 layers: 30 linear-attention (GatedDeltaNet/ArraysCache) + 10 full-attention (Attention/KVCache), every 4th layer is full-attention
- Quantization read from config.json (bits + group_size). Supports 4-bit and 8-bit.
- Expert dimensions: hidden=2048, intermediate=512, 256 experts/layer, top_k=8
- **4-bit (group_size=64)**: per_expert_stride ~1.69 MB, ECB file ~453 MB/layer, 8 active = ~13.5 MB/layer
- **8-bit (group_size=32)**: per_expert_stride 3.38 MB, ECB file 906 MB/layer, 8 active = 27 MB/layer
- 12 CPU threads available (M4)

### Current model: 4-bit
- Path: `/Users/philtrem/.lmstudio/models/mlx-community/Qwen3.5-35B-A3B-4bit`
- Split output: `./split_model_st` (19 GB)
- Warm set backup: `~/warm_experts_qwen35_35b_a3b.json` (from 8-bit profiling, compatible with 4-bit)

## Performance

### Current best — 4-bit, reactive pread + warm set (no flags):
- **50 tokens**: **7.5 tok/s** (implied 7.9), ramping to 8.4
- Decode breakdown (127ms/tok): routing CPU/pread 55ms (43%), layer eval 32ms (25%), routing eval 30ms (24%), extract 11ms (9%)
- GPU wait: 26ms (pure compute, zero faults)
- Blocking pread ensures fault-free eval. Warm set cuts pread from 75ms to 55ms.

### 4-bit benchmark matrix (50 tokens, M4 16GB):
| Config | tok/s | Pread ms/tok | Layer eval ms/tok | GPU wait ms/tok |
|--------|-------|-------------|-------------------|-----------------|
| **Reactive + warm set** | **7.5→8.4** | **54.6** | **31.5** | **26.2** |
| Reactive only, no warm set | 6.1→8.1 | 75.1 | 33.1 | 26.8 |
| Reactive + F_RDADVISE spec | 6.9→7.6 | 62.4 | 31.1 | 16.4 |
| Warm set only (no reactive) | 5.3→6.6 | 0.1 | 137.3 | 132.5 |
| Reactive + main-thread pread spec | 4.7 | 33.7 | 112.7 | 1.0 |
| Reactive + POSIX AIO spec | 4.7 | 34.8 | 112.5 | 34.0 |
| Reactive + bg rayon spec (pre-alloc) | 4.5 | 35.4 | 118.1 | 33.8 |
| Reactive + bg rayon spec (4-thread) | 5.3 | 42.3 | 94.3 | 29.6 |
| Reactive + mincore+pread spec | 4.3 | 34.3 | 131.5 | 1.3 |
| Reactive + madvise spec | 4.2 | 34.2 | 139.2 | 3.0 |
| Speculative only (no reactive) | 3.9 | 0.1 | 195.7 | 130.4 |

### Why speculative fails on 4-bit
- Experts are 1.69 MB (small) — page cache retains them between tokens
- GPU eval is ~0.65ms/layer — too short for speculative I/O to overlap meaningfully
- ANY speculative overhead (even F_RDADVISE at ~0.5ms/tok) creates SSD contention with reactive pread
- Speculative preading 12 experts to save on 8 reactive preads reads 50% more data than needed
- Background par_iter is 2.3× slower than main-thread par_iter on macOS
- Main-thread speculative overlaps GPU perfectly (GPU wait→1ms) but blocks the pipeline for 2.8ms/layer

### 8-bit historical (Qwen3.5-35B-A3B-MLX-9bit, now deleted):
- Default mmap zerocopy: 3.8 tok/s (warm cache), 1.8-2.0 tok/s (cold/degraded)
- Reactive pread + no warm set: 2.8 tok/s (8-bit experts too large for page cache on 16 GB)
- async_eval pipeline: 1.8-2.1 tok/s (pread throughput bottleneck)
- Pure GPU compute: ~57ms/tok (fault-free eval)

### SSD benchmarks (M4 Mac Mini base):
- Actual SSD read rate: ~3 GB/s
- Random 16 KB read (cold): 153 MB/s (107μs per read)
- Page faults: 16KB per fault, synchronous kernel trap — 20× slower per byte than explicit pread

### Expert prediction (measured 2026-03-30):
- **Pre-MoE + next LN, top-12**: 84-86% accuracy (stable across runs)
- Per-layer range: 57% (layer 1) to 96% (layer 34)
- Prediction adds ~6ms routing eval overhead (batched into existing eval)

## Key gotchas

### MLX / mlx-rs
- `mlx_array_new_data` (and `Array::from_raw_data`) **copies** data — use `ffi_zerocopy.cpp` shim for zero-copy via `newBufferWithBytesNoCopy`
- `Array::load_safetensors()` creates lazy arrays; loading all expert files causes swap storms on 16 GB
- `gather_qmm` is NOT in mlx-rs — use `mlx_sys::mlx_gather_qmm` via FFI wrapper
- `argsort` runs on CPU; eval boundaries needed before GPU gather_qmm
- Activation dtype drift: bf16×f32 scalar promotes to f32 — cast scalars to input dtype
- `array_from_mmap` (Metal buffer creation via newBufferWithBytesNoCopy) is fast: 0.3ms for 108 arrays. NOT a bottleneck.

### Memory / UMA
- **Do NOT load all expert files via load_safetensors** — causes swap storms on 16 GB
- On-demand expert extraction via pread() is the correct approach
- **pread() is 3.6× faster than mmap demand-paging** (page fault overhead: ~107μs/page for cold 16 KB random reads)
- **Blocking pread before eval is the winning strategy**: pread at contiguous MB granularity warms page cache; subsequent mmap zerocopy eval runs fault-free. Saves 100-200ms/tok vs page faults during eval.
- **madvise(MADV_WILLNEED) is unreliable** — returns before pages are loaded. Use pread to guarantee page residency.
- **mlock HURTS**: page table wire/unwire contends with GPU at kernel vm_map level.
- **Speculative prefetch is a net negative on 4-bit** in ALL tested variants: pread (background pool, main thread, POSIX AIO, pre-alloc buffer), F_RDADVISE, madvise, mincore+pread. Every method either wastes time on already-cached pages or creates SSD contention with reactive pread. See benchmark matrix.
- **F_RDADVISE** is the lightest speculative method (12 fcntl calls, ~0.5ms/tok) but still causes SSD contention with reactive pread (+8ms). Net negative on 4-bit. May help on 8-bit where experts are larger and GPU eval is longer.
- Per-layer eval ensures expert arrays are freed after each layer (peak ~13.5 MB for 4-bit, not cumulative)
- Expert LRU caching does NOT help — working set >> cache size on 16 GB
- **Warm set pread at startup** (not madvise): guarantees 63% of expert pages are resident. Saves ~20ms/tok on reactive pread.

### I/O architecture findings (2026-03-29/30):
- **Page faults as flow control**: GPU self-throttles to match SSD throughput. Natural pipelining — but explicit pread is more efficient per byte.
- **Speculative prefetch during eval causes SSD contention**: preads for L+1 compete with L's page faults or L+1's reactive pread. Consistently worse in all tested configurations.
- **Background par_iter is ~2.3× slower than main-thread par_iter**. Cause unknown — likely SSD/kernel-level scheduling.
- **async_eval pipeline** (explicit pread + async_eval overlap): I/O-compute overlap confirmed working (eval_wait=0ms). But pread throughput is the bottleneck, not overlap.
- **The 2.7× pread gap** (8-bit): measured 400ms vs 150ms theoretical. Root cause: page cache can't hold warm set on 16 GB under memory pressure from 8-bit expert files. NOT per-call overhead (wrapping is 0.3ms for 108 Metal buffers).

### Env vars for testing:
- `NOREACTIVE=1` — skip reactive blocking pread (for A/B testing)

### Speculative prefetch methods tested (all net negative on 4-bit):
- **F_RDADVISE**: lightest (fcntl hint, ~0.5ms/tok). SSD contention with reactive.
- **Main-thread pread**: perfect GPU overlap (wait=1ms) but 2.8ms/layer pread blocks pipeline.
- **Background rayon pool (4-thread)**: 2.3× background-thread penalty on macOS.
- **Background rayon (pre-alloc buffer)**: eliminating alloc didn't help — penalty is thread-level.
- **Background rayon (global pool, 12-thread)**: CPU oversubscription (24 threads on 12 cores).
- **POSIX AIO (aio_read)**: macOS emulates with kernel threads — same overhead as rayon.
- **madvise(MADV_WILLNEED)**: slow page-table walks (3.4ms/layer). Worse than pread.
- **mincore + pread (cold-only)**: mincore check overhead (480 syscalls) exceeds savings.

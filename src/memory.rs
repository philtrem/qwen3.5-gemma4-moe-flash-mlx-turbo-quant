use std::collections::HashSet;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::os::unix::io::RawFd;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;

use memmap2::Mmap;
use mlx_rs::{Array, Dtype};
use rayon::prelude::*;
use serde_json::Value;

// ── Safetensors format structs ──────────────────────────────────────────────

/// Per-tensor offset info parsed from a safetensors header.
struct TensorInfo {
    /// Byte offset of this tensor's data from `data_start`
    data_offset: usize,
    /// Bytes per single expert slice
    per_expert_stride: usize,
    /// Per-expert shape (shape[1:]), e.g. [512, 512] for gate_proj.weight
    expert_shape: Vec<i32>,
    /// MLX dtype
    dtype: Dtype,
}

/// Parsed safetensors layout for one layer file.
struct LayerTensorOffsets {
    /// Start of tensor data: 8 + header_size
    data_start: usize,
    /// The 9 expert tensors in order
    tensors: Vec<TensorInfo>,
}

// ── ECB format structs ──────────────────────────────────────────────────────

/// Per-tensor descriptor parsed from ECB header.
struct EcbTensorDesc {
    /// Byte offset of this tensor within each expert's contiguous block
    offset_within_expert: usize,
    /// Bytes per expert for this tensor
    stride: usize,
    /// Per-expert shape (excludes expert dimension)
    expert_shape: Vec<i32>,
    /// MLX dtype
    dtype: Dtype,
}

/// Parsed ECB layout for one layer file.
struct EcbLayerInfo {
    /// Byte offset where expert data begins (= header_size, page-aligned)
    data_start: usize,
    /// Total bytes per expert (sum of all tensor strides)
    per_expert_stride: usize,
    /// The 9 tensor descriptors
    tensors: Vec<EcbTensorDesc>,
}

/// Which expert file format is in use.
enum ExpertFormat {
    Safetensors(Vec<LayerTensorOffsets>),
    Ecb(Vec<EcbLayerInfo>),
}

// ── Shared types ────────────────────────────────────────────────────────────

/// The 9 expert arrays extracted for a set of active experts.
/// Each array has shape [num_experts, d1, d2].
pub struct ExpertSlice {
    pub gate_weight: Array,
    pub gate_scales: Array,
    pub gate_biases: Array,
    pub up_weight: Array,
    pub up_scales: Array,
    pub up_biases: Array,
    pub down_weight: Array,
    pub down_scales: Array,
    pub down_biases: Array,
}

impl ExpertSlice {
    /// Construct from a Vec of exactly 9 arrays in gate/up/down × weight/scales/biases order.
    fn from_arrays(mut arrays: Vec<Array>) -> Self {
        debug_assert_eq!(arrays.len(), 9);
        Self {
            gate_weight: arrays.remove(0),
            gate_scales: arrays.remove(0),
            gate_biases: arrays.remove(0),
            up_weight: arrays.remove(0),
            up_scales: arrays.remove(0),
            up_biases: arrays.remove(0),
            down_weight: arrays.remove(0),
            down_scales: arrays.remove(0),
            down_biases: arrays.remove(0),
        }
    }
}

/// A single expert's 9 tensors (not stacked across experts).
/// Each tensor has shape [d1, d2] — for per-expert quantized_matmul.
pub struct SingleExpertTensors {
    pub gate_weight: Array,
    pub gate_scales: Array,
    pub gate_biases: Array,
    pub up_weight: Array,
    pub up_scales: Array,
    pub up_biases: Array,
    pub down_weight: Array,
    pub down_scales: Array,
    pub down_biases: Array,
}

// ── Async I/O prefetch ─────────────────────────────────────────────────────

/// Per-layer info passed to the I/O prefetch thread (all Send-safe).
struct LayerIoInfo {
    fd: RawFd,
    data_start: usize,
    per_expert_stride: usize,
}

/// Work item for the I/O prefetch thread.
struct PrefetchWork {
    layer: usize,
    expert_indices: Vec<i32>,
}

/// Manages expert files with direct pread() extraction.
///
/// Supports both safetensors (72 preads/layer) and ECB (8 preads/layer) formats.
/// ECB is auto-detected by probing for .ecb files; falls back to safetensors.
#[allow(dead_code)]
pub struct ExpertMemoryManager {
    files: Vec<File>,       // for pread() extraction
    maps: Vec<Mmap>,        // for warm set madvise only
    format: ExpertFormat,
    warm_set: HashSet<(u32, u32)>,
    hits: AtomicUsize,
    misses: AtomicUsize,
    // Pre-allocated page-aligned buffer for hybrid pread+zerocopy path.
    hybrid_buf_ptr: *mut u8,
    hybrid_buf_layout: std::alloc::Layout,
    // Async I/O prefetch thread — reads expert data to force pages into cache.
    io_tx: Option<mpsc::Sender<Option<PrefetchWork>>>,
    io_done_rx: Option<mpsc::Receiver<()>>,
    io_handle: Option<thread::JoinHandle<()>>,
}

// ── F_RDADVISE FFI (macOS) ──────────────────────────────────────────────────

#[repr(C)]
struct Radvisory {
    ra_offset: libc::off_t,
    ra_count: libc::c_int,
}

const F_RDADVISE: libc::c_int = 44;

fn issue_rdadvise(fd: std::os::unix::io::RawFd, offset: usize, len: usize) {
    let mut advice = Radvisory {
        ra_offset: offset as libc::off_t,
        ra_count: len as libc::c_int,
    };
    unsafe {
        libc::fcntl(fd, F_RDADVISE, &mut advice);
    }
}

/// Page-aligned madvise(MADV_WILLNEED). Returns bytes advised.
fn advise_willneed(mmap: &Mmap, abs_start: usize, len: usize) -> usize {
    const PAGE_SIZE: usize = 16384; // Apple Silicon
    let aligned_start = abs_start & !(PAGE_SIZE - 1);
    let aligned_len = (abs_start + len - aligned_start + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
    unsafe {
        let ptr = mmap.as_ptr().add(aligned_start);
        libc::madvise(ptr as *mut _, aligned_len, libc::MADV_WILLNEED);
    }
    aligned_len
}

// ── Format parsing ──────────────────────────────────────────────────────────

fn safetensors_dtype_to_mlx(dtype_str: &str) -> Dtype {
    match dtype_str {
        "U32" => Dtype::Uint32,
        "BF16" => Dtype::Bfloat16,
        "F16" => Dtype::Float16,
        "F32" => Dtype::Float32,
        "I32" => Dtype::Int32,
        "U8" => Dtype::Uint8,
        _ => panic!("unsupported safetensors dtype: {}", dtype_str),
    }
}

fn ecb_dtype_to_mlx(code: u32) -> Dtype {
    match code {
        0 => Dtype::Uint8,
        1 => Dtype::Uint32,
        2 => Dtype::Bfloat16,
        3 => Dtype::Float16,
        4 => Dtype::Float32,
        5 => Dtype::Int32,
        _ => panic!("unsupported ECB dtype code: {}", code),
    }
}

/// Parse a safetensors header to extract per-tensor byte offsets, strides, shapes, and dtypes.
fn parse_layer_offsets(mmap: &[u8]) -> anyhow::Result<LayerTensorOffsets> {
    let header_size = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
    let data_start = 8 + header_size;
    let header: Value = serde_json::from_slice(&mmap[8..data_start])?;

    let tensor_names = [
        "gate_proj.weight", "gate_proj.scales", "gate_proj.biases",
        "up_proj.weight",   "up_proj.scales",   "up_proj.biases",
        "down_proj.weight", "down_proj.scales",  "down_proj.biases",
    ];

    let mut tensors = Vec::with_capacity(9);
    for name in &tensor_names {
        let info = header.get(*name)
            .ok_or_else(|| anyhow::anyhow!("missing tensor {} in safetensors header", name))?;

        let data_offsets = info["data_offsets"].as_array()
            .ok_or_else(|| anyhow::anyhow!("no data_offsets for {}", name))?;
        let start = data_offsets[0].as_u64().unwrap() as usize;
        let end = data_offsets[1].as_u64().unwrap() as usize;

        let shape: Vec<usize> = info["shape"].as_array().unwrap()
            .iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let dtype_str = info["dtype"].as_str().unwrap();
        let dtype = safetensors_dtype_to_mlx(dtype_str);
        let num_experts = shape[0];
        let expert_shape: Vec<i32> = shape[1..].iter().map(|&s| s as i32).collect();

        let total_bytes = end - start;
        let per_expert_stride = total_bytes / num_experts;

        tensors.push(TensorInfo {
            data_offset: start,
            per_expert_stride,
            expert_shape,
            dtype,
        });
    }

    Ok(LayerTensorOffsets { data_start, tensors })
}

/// Parse an ECB header (first 16384 bytes) to extract tensor descriptors.
fn parse_ecb_header(file: &File) -> anyhow::Result<EcbLayerInfo> {
    // Read the first 20 bytes to get fixed fields
    let mut fixed = [0u8; 20];
    file.read_exact_at(&mut fixed, 0)?;

    let magic = &fixed[0..4];
    if magic != b"ECB1" {
        anyhow::bail!("not an ECB file (bad magic)");
    }

    let _num_experts = u32::from_le_bytes(fixed[4..8].try_into().unwrap());
    let per_expert_stride = u32::from_le_bytes(fixed[8..12].try_into().unwrap()) as usize;
    let num_tensors = u32::from_le_bytes(fixed[12..16].try_into().unwrap()) as usize;
    let header_size = u32::from_le_bytes(fixed[16..20].try_into().unwrap()) as usize;

    // Read remaining header bytes for tensor descriptors
    let desc_bytes_needed = header_size - 20;
    let mut desc_buf = vec![0u8; desc_bytes_needed];
    file.read_exact_at(&mut desc_buf, 20)?;

    let mut tensors = Vec::with_capacity(num_tensors);
    let mut pos = 0usize;
    let mut cumulative_offset = 0usize;

    for _ in 0..num_tensors {
        let stride = u32::from_le_bytes(desc_buf[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let dtype_code = u32::from_le_bytes(desc_buf[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let ndim = u32::from_le_bytes(desc_buf[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        let mut expert_shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            let dim = u32::from_le_bytes(desc_buf[pos..pos + 4].try_into().unwrap()) as i32;
            pos += 4;
            expert_shape.push(dim);
        }

        tensors.push(EcbTensorDesc {
            offset_within_expert: cumulative_offset,
            stride,
            expert_shape,
            dtype: ecb_dtype_to_mlx(dtype_code),
        });
        cumulative_offset += stride;
    }

    Ok(EcbLayerInfo {
        data_start: header_size,
        per_expert_stride,
        tensors,
    })
}

// ── ExpertMemoryManager ─────────────────────────────────────────────────────

impl ExpertMemoryManager {
    /// Open expert files: auto-detects ECB vs safetensors format.
    /// mmap for headers + madvise, File handles for pread.
    pub fn new(expert_dir: &Path, num_layers: usize) -> anyhow::Result<Self> {
        // Auto-detect format: try ECB first
        let ecb_probe = expert_dir.join("layer_00_experts.ecb");
        let use_ecb = ecb_probe.exists();

        let ext = if use_ecb { "ecb" } else { "safetensors" };
        eprintln!("  Expert format: {}", ext);

        let mut files = Vec::with_capacity(num_layers);
        let mut maps = Vec::with_capacity(num_layers);

        if use_ecb {
            let mut ecb_infos = Vec::with_capacity(num_layers);
            for i in 0..num_layers {
                let path = expert_dir.join(format!("layer_{:02}_experts.ecb", i));
                let pread_file = File::open(&path)?;
                let info = parse_ecb_header(&pread_file)?;
                ecb_infos.push(info);
                // mmap for warm set madvise (separate fd)
                let mmap_file = File::open(&path)?;
                let mmap = unsafe { Mmap::map(&mmap_file)? };
                maps.push(mmap);
                files.push(pread_file);
            }
            // Allocate page-aligned buffer for hybrid pread+zerocopy path
            let max_stride = ecb_infos.iter().map(|i| i.per_expert_stride).max().unwrap_or(0);
            let hybrid_buf_size = 16 * max_stride;
            let hybrid_buf_layout = std::alloc::Layout::from_size_align(hybrid_buf_size, 16384)
                .expect("invalid layout for hybrid buffer");
            let hybrid_buf_ptr = unsafe { std::alloc::alloc(hybrid_buf_layout) };
            if hybrid_buf_ptr.is_null() {
                std::alloc::handle_alloc_error(hybrid_buf_layout);
            }

            // Spawn async I/O prefetch thread
            let layer_infos: Vec<LayerIoInfo> = files.iter().zip(ecb_infos.iter())
                .map(|(f, info)| LayerIoInfo {
                    fd: f.as_raw_fd(),
                    data_start: info.data_start,
                    per_expert_stride: info.per_expert_stride,
                })
                .collect();

            let (io_tx, io_rx) = mpsc::channel::<Option<PrefetchWork>>();
            let (io_done_tx, io_done_rx) = mpsc::channel::<()>();

            let io_handle = thread::Builder::new()
                .name("expert-prefetch".into())
                .spawn(move || {
                    let max_stride = layer_infos.iter()
                        .map(|i| i.per_expert_stride).max().unwrap_or(0);
                    let mut buf = vec![0u8; max_stride];

                    while let Ok(Some(work)) = io_rx.recv() {
                        let info = &layer_infos[work.layer];
                        for &eidx in &work.expert_indices {
                            let offset = info.data_start as i64
                                + eidx as i64 * info.per_expert_stride as i64;
                            // pread into throwaway buffer — blocks until pages
                            // are in kernel page cache. The blocking is the
                            // guarantee: when pread returns, pages are resident.
                            unsafe {
                                libc::pread(
                                    info.fd,
                                    buf.as_mut_ptr() as *mut libc::c_void,
                                    info.per_expert_stride,
                                    offset,
                                );
                            }
                        }
                        let _ = io_done_tx.send(());
                    }
                })
                .expect("failed to spawn prefetch thread");

            eprintln!("  I/O prefetch thread: started");

            Ok(Self {
                files,
                maps,
                format: ExpertFormat::Ecb(ecb_infos),
                warm_set: HashSet::new(),
                hits: AtomicUsize::new(0),
                misses: AtomicUsize::new(0),
                hybrid_buf_ptr,
                hybrid_buf_layout,
                io_tx: Some(io_tx),
                io_done_rx: Some(io_done_rx),
                io_handle: Some(io_handle),
            })
        } else {
            let mut st_offsets = Vec::with_capacity(num_layers);
            for i in 0..num_layers {
                let path = expert_dir.join(format!("layer_{:02}_experts.safetensors", i));
                let file = File::open(&path)?;
                let mmap = unsafe { Mmap::map(&file)? };
                let layer_offsets = parse_layer_offsets(&mmap)?;
                st_offsets.push(layer_offsets);
                maps.push(mmap);
                files.push(File::open(&path)?);
            }
            Ok(Self {
                files,
                maps,
                format: ExpertFormat::Safetensors(st_offsets),
                warm_set: HashSet::new(),
                hits: AtomicUsize::new(0),
                misses: AtomicUsize::new(0),
                hybrid_buf_ptr: std::ptr::null_mut(),
                hybrid_buf_layout: std::alloc::Layout::from_size_align(1, 1).unwrap(),
                io_tx: None,
                io_done_rx: None,
                io_handle: None,
            })
        }
    }

    /// Record the warm set for hit rate tracking.
    pub fn set_warm_set(&mut self, experts: &[(u32, u32)]) {
        self.warm_set = experts.iter().copied().collect();
    }

    /// Return (hits, misses, hit_rate). Resets counters.
    pub fn take_hit_stats(&self) -> (usize, usize, f64) {
        let h = self.hits.swap(0, Ordering::Relaxed);
        let m = self.misses.swap(0, Ordering::Relaxed);
        let rate = if h + m > 0 { h as f64 / (h + m) as f64 } else { 0.0 };
        (h, m, rate)
    }

    /// Dummy methods for compatibility with engine.rs cache reporting
    pub fn take_cache_stats(&self) -> (u64, u64, f64) { (0, 0, 0.0) }
    pub fn reset_cache_stats(&self) {}
    pub fn cache_size(&self) -> usize { 0 }

    /// Signal the background I/O thread to prefetch experts for a layer.
    /// Non-blocking: sends work to the channel and returns immediately.
    /// The I/O thread reads expert data via pread, populating the kernel
    /// page cache. On UMA, these pages are the same physical memory that
    /// backs the mmap'd Metal buffers — no copy to GPU needed.
    pub fn prefetch_async(&self, layer: usize, expert_indices: &[i32]) {
        if let Some(tx) = &self.io_tx {
            let _ = tx.send(Some(PrefetchWork {
                layer,
                expert_indices: expert_indices.to_vec(),
            }));
        }
    }

    #[allow(dead_code)]
    pub fn wait_for_prefetch(&self) {
        if let Some(rx) = &self.io_done_rx {
            let _ = rx.recv();
        }
    }

    /// Partition expert indices into (warm, cold) based on the static warm set.
    /// Warm experts are likely in page cache (free via mmap). Cold experts need
    /// SSD reads — the I/O thread should prioritize these.
    pub fn partition_warm_cold(&self, layer: usize, expert_indices: &[i32]) -> (Vec<i32>, Vec<i32>) {
        let mut warm = Vec::new();
        let mut cold = Vec::new();
        for &eidx in expert_indices {
            if self.warm_set.contains(&(layer as u32, eidx as u32)) {
                warm.push(eidx);
            } else {
                cold.push(eidx);
            }
        }
        (warm, cold)
    }

    /// Extract specific experts from a layer.
    /// Dispatches to ECB (8 parallel preads) or safetensors (72 preads) path.
    pub fn extract_experts(&self, layer: usize, expert_indices: &[i32]) -> ExpertSlice {
        // Track warm set hits
        for &eidx in expert_indices {
            if self.warm_set.contains(&(layer as u32, eidx as u32)) {
                self.hits.fetch_add(1, Ordering::Relaxed);
            } else {
                self.misses.fetch_add(1, Ordering::Relaxed);
            }
        }

        match &self.format {
            ExpertFormat::Ecb(infos) => self.extract_experts_ecb(&infos[layer], layer, expert_indices),
            ExpertFormat::Safetensors(offsets) => self.extract_experts_safetensors(&offsets[layer], layer, expert_indices),
        }
    }

    /// ECB extract: 8 parallel preads (one per expert, ~3.375 MB each),
    /// then scatter into 9 per-tensor arrays for gather_qmm.
    fn extract_experts_ecb(&self, info: &EcbLayerInfo, layer: usize, expert_indices: &[i32]) -> ExpertSlice {
        let file = &self.files[layer];
        let stride = info.per_expert_stride;
        let n = expert_indices.len();

        // Parallel pread: one large contiguous read per expert.
        // Uninitialized buffers — pread overwrites every byte.
        let expert_bufs: Vec<Vec<u8>> = expert_indices.par_iter()
            .map(|&eidx| {
                let mut buf = Vec::with_capacity(stride);
                unsafe { buf.set_len(stride); }
                let offset = info.data_start as u64 + eidx as u64 * stride as u64;
                file.read_exact_at(&mut buf, offset).expect("pread failed");
                buf
            })
            .collect();

        // Scatter into 9 per-tensor buffers via direct indexing (no extend_from_slice)
        let mut arrays = Vec::with_capacity(9);
        for tensor in &info.tensors {
            let t_stride = tensor.stride;
            let total = n * t_stride;
            let mut tensor_buf = Vec::with_capacity(total);
            unsafe { tensor_buf.set_len(total); }

            for (i, expert_buf) in expert_bufs.iter().enumerate() {
                tensor_buf[i * t_stride..(i + 1) * t_stride]
                    .copy_from_slice(&expert_buf[tensor.offset_within_expert..tensor.offset_within_expert + t_stride]);
            }

            let mut shape = vec![n as i32];
            shape.extend_from_slice(&tensor.expert_shape);
            let arr = unsafe {
                Array::from_raw_data(
                    tensor_buf.as_ptr() as *const std::ffi::c_void,
                    &shape,
                    tensor.dtype,
                )
            };
            arrays.push(arr);
        }

        ExpertSlice::from_arrays(arrays)
    }

    /// Safetensors extract: 72 preads per layer (9 tensors × 8 experts).
    /// Kept as fallback for backward compatibility.
    fn extract_experts_safetensors(&self, layer_offsets: &LayerTensorOffsets, layer: usize, expert_indices: &[i32]) -> ExpertSlice {
        let file = &self.files[layer];
        let n = expert_indices.len() as i32;

        let mut arrays = Vec::with_capacity(9);
        for tensor in &layer_offsets.tensors {
            let stride = tensor.per_expert_stride;
            let total = expert_indices.len() * stride;
            let mut buf = vec![0u8; total];

            for (i, &eidx) in expert_indices.iter().enumerate() {
                let file_offset = (layer_offsets.data_start
                    + tensor.data_offset
                    + eidx as usize * stride) as u64;
                file.read_exact_at(
                    &mut buf[i * stride..(i + 1) * stride],
                    file_offset,
                )
                .expect("pread failed");
            }

            let mut shape = vec![n];
            shape.extend_from_slice(&tensor.expert_shape);
            let arr = unsafe {
                Array::from_raw_data(
                    buf.as_ptr() as *const std::ffi::c_void,
                    &shape,
                    tensor.dtype,
                )
            };
            arrays.push(arr);
        }

        ExpertSlice::from_arrays(arrays)
    }

    /// Zero-copy extract: create MLX arrays backed directly by mmap'd memory.
    /// Requires ECB format (expert data is contiguous and page-aligned).
    /// Returns per-expert tensors for use with quantized_matmul (not gather_qmm).
    pub fn extract_expert_zerocopy(&self, layer: usize, expert_idx: i32) -> SingleExpertTensors {
        let mmap = &self.maps[layer];
        let info = match &self.format {
            ExpertFormat::Ecb(infos) => &infos[layer],
            ExpertFormat::Safetensors(_) => panic!("zero-copy requires ECB format"),
        };

        let expert_offset = info.data_start + expert_idx as usize * info.per_expert_stride;
        let mmap_ptr = mmap.as_ptr();

        let mut arrays = Vec::with_capacity(9);
        for tensor in &info.tensors {
            let tensor_offset = expert_offset + tensor.offset_within_expert;
            let arr = unsafe {
                crate::ffi::array_from_mmap(
                    mmap_ptr,
                    tensor_offset,
                    tensor.stride,
                    &tensor.expert_shape,
                    tensor.dtype,
                )
            };
            arrays.push(arr);
        }

        SingleExpertTensors {
            gate_weight: arrays.remove(0),
            gate_scales: arrays.remove(0),
            gate_biases: arrays.remove(0),
            up_weight: arrays.remove(0),
            up_scales: arrays.remove(0),
            up_biases: arrays.remove(0),
            down_weight: arrays.remove(0),
            down_scales: arrays.remove(0),
            down_biases: arrays.remove(0),
        }
    }

    /// Issue F_RDADVISE for the next layer's expected expert regions.
    /// Non-blocking — kernel reads asynchronously while GPU processes current layer.
    /// Exploits expert locality across adjacent layers.
    pub fn prefetch_next_layer(&self, current_layer: usize, expert_indices: &[i32]) {
        let next_layer = current_layer + 1;
        if next_layer >= self.files.len() {
            return;
        }

        let fd = self.files[next_layer].as_raw_fd();

        match &self.format {
            ExpertFormat::Ecb(infos) => {
                let info = &infos[next_layer];
                let stride = info.per_expert_stride;
                for &eidx in expert_indices {
                    issue_rdadvise(fd, info.data_start + eidx as usize * stride, stride);
                }
            }
            ExpertFormat::Safetensors(offsets) => {
                let layer_offsets = &offsets[next_layer];
                for &eidx in expert_indices {
                    for tensor in &layer_offsets.tensors {
                        let offset = layer_offsets.data_start
                            + tensor.data_offset
                            + eidx as usize * tensor.per_expert_stride;
                        issue_rdadvise(fd, offset, tensor.per_expert_stride);
                    }
                }
            }
        }
    }

    /// Issue F_RDADVISE for specific experts at a given layer.
    #[allow(dead_code)]
    pub fn prefetch_experts(&self, layer: usize, expert_indices: &[i32]) {
        if layer >= self.files.len() {
            return;
        }
        let fd = self.files[layer].as_raw_fd();
        match &self.format {
            ExpertFormat::Ecb(infos) => {
                let info = &infos[layer];
                let stride = info.per_expert_stride;
                for &eidx in expert_indices {
                    issue_rdadvise(fd, info.data_start + eidx as usize * stride, stride);
                }
            }
            ExpertFormat::Safetensors(offsets) => {
                let layer_offsets = &offsets[layer];
                for &eidx in expert_indices {
                    for tensor in &layer_offsets.tensors {
                        let offset = layer_offsets.data_start
                            + tensor.data_offset
                            + eidx as usize * tensor.per_expert_stride;
                        issue_rdadvise(fd, offset, tensor.per_expert_stride);
                    }
                }
            }
        }
    }

    #[allow(dead_code)]
    pub fn prefetch_experts_madvise(&self, layer: usize, expert_indices: &[i32]) {
        if layer >= self.maps.len() {
            return;
        }
        let mmap = &self.maps[layer];
        match &self.format {
            ExpertFormat::Ecb(infos) => {
                let info = &infos[layer];
                for &eidx in expert_indices {
                    let abs_start = info.data_start + eidx as usize * info.per_expert_stride;
                    advise_willneed(mmap, abs_start, info.per_expert_stride);
                }
            }
            ExpertFormat::Safetensors(offsets) => {
                let layer_offsets = &offsets[layer];
                for &eidx in expert_indices {
                    for tensor in &layer_offsets.tensors {
                        let offset = layer_offsets.data_start
                            + tensor.data_offset
                            + eidx as usize * tensor.per_expert_stride;
                        advise_willneed(mmap, offset, tensor.per_expert_stride);
                    }
                }
            }
        }
    }

    /// Hybrid pread+zerocopy: parallel pread into pre-allocated buffer,
    /// then wrap slices as zero-copy Metal arrays via newBufferWithBytesNoCopy.
    ///
    /// Eliminates page faults entirely: explicit I/O at high NVMe queue depth
    /// (one pread per expert in parallel via rayon) followed by zero-copy GPU access.
    ///
    #[allow(dead_code)]
    pub fn extract_experts_hybrid(&self, layer: usize, expert_indices: &[i32]) -> Vec<SingleExpertTensors> {
        let info = match &self.format {
            ExpertFormat::Ecb(infos) => &infos[layer],
            ExpertFormat::Safetensors(_) => panic!("hybrid requires ECB format"),
        };
        let n = expert_indices.len();
        let stride = info.per_expert_stride;
        let total = n * stride;
        assert!(
            total <= self.hybrid_buf_layout.size(),
            "hybrid buffer too small: need {} bytes for {} experts",
            total, n
        );

        let file = &self.files[layer];
        let buf_addr = self.hybrid_buf_ptr as usize;
        let data_start = info.data_start;

        // Parallel pread: one contiguous read per expert, high NVMe queue depth.
        // Each thread writes to a non-overlapping region of the buffer.
        expert_indices.par_iter().enumerate().for_each(|(i, &eidx)| {
            let file_offset = data_start as u64 + eidx as u64 * stride as u64;
            let dst = unsafe {
                std::slice::from_raw_parts_mut((buf_addr + i * stride) as *mut u8, stride)
            };
            file.read_exact_at(dst, file_offset).expect("pread failed");
        });

        // Wrap buffer slices as zero-copy Metal arrays (no data copy, no page faults)
        let buf_ptr = self.hybrid_buf_ptr;
        expert_indices.iter().enumerate().map(|(i, _)| {
            let expert_base = i * stride;
            let mut arrays = Vec::with_capacity(9);
            for tensor in &info.tensors {
                let offset = expert_base + tensor.offset_within_expert;
                let arr = unsafe {
                    crate::ffi::array_from_mmap(
                        buf_ptr,
                        offset,
                        tensor.stride,
                        &tensor.expert_shape,
                        tensor.dtype,
                    )
                };
                arrays.push(arr);
            }
            SingleExpertTensors {
                gate_weight: arrays.remove(0),
                gate_scales: arrays.remove(0),
                gate_biases: arrays.remove(0),
                up_weight: arrays.remove(0),
                up_scales: arrays.remove(0),
                up_biases: arrays.remove(0),
                down_weight: arrays.remove(0),
                down_scales: arrays.remove(0),
                down_biases: arrays.remove(0),
            }
        }).collect()
    }

    /// Prefetch warm set expert pages into kernel page cache via madvise.
    /// Returns total bytes advised.
    pub fn mlock_warm_set(&self, experts: &[(u32, u32)]) -> usize {
        let mut advised = 0usize;

        for &(layer, expert_idx) in experts {
            let layer = layer as usize;
            let expert_idx = expert_idx as usize;

            if layer >= self.maps.len() {
                continue;
            }
            let mmap = &self.maps[layer];

            match &self.format {
                ExpertFormat::Ecb(infos) => {
                    let info = &infos[layer];
                    let abs_start = info.data_start + expert_idx * info.per_expert_stride;
                    advised += advise_willneed(mmap, abs_start, info.per_expert_stride);
                }
                ExpertFormat::Safetensors(offsets) => {
                    let layer_offsets = &offsets[layer];
                    for tensor in &layer_offsets.tensors {
                        let abs_start = layer_offsets.data_start
                            + tensor.data_offset
                            + expert_idx * tensor.per_expert_stride;
                        advised += advise_willneed(mmap, abs_start, tensor.per_expert_stride);
                    }
                }
            }
        }

        advised
    }
}

impl Drop for ExpertMemoryManager {
    fn drop(&mut self) {
        // Shut down I/O thread BEFORE files close (fds must remain valid).
        if let Some(tx) = self.io_tx.take() {
            let _ = tx.send(None); // signal shutdown
        }
        if let Some(handle) = self.io_handle.take() {
            let _ = handle.join();
        }
        if !self.hybrid_buf_ptr.is_null() {
            unsafe { std::alloc::dealloc(self.hybrid_buf_ptr, self.hybrid_buf_layout); }
        }
    }
}

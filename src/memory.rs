use std::path::Path;

use memmap2::Mmap;
use mlx_rs::{Array, Dtype};
use serde_json::Value;

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
    /// The 9 expert tensors in order:
    /// gate_proj.{weight,scales,biases}, up_proj.{weight,scales,biases}, down_proj.{weight,scales,biases}
    tensors: Vec<TensorInfo>,
}

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

/// Manages mmap'd expert safetensors files.
/// Provides on-demand extraction of expert slices for MoE computation.
pub struct ExpertMemoryManager {
    maps: Vec<Mmap>,
    offsets: Vec<LayerTensorOffsets>,
}

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
        let num_experts = shape[0]; // always 256
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

impl ExpertMemoryManager {
    /// Open and mmap all expert safetensors files, parse headers.
    pub fn new(expert_dir: &Path, num_layers: usize) -> anyhow::Result<Self> {
        let mut maps = Vec::with_capacity(num_layers);
        let mut offsets = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let path = expert_dir.join(format!("layer_{:02}_experts.safetensors", i));
            let file = std::fs::File::open(&path)?;
            let mmap = unsafe { Mmap::map(&file)? };
            let layer_offsets = parse_layer_offsets(&mmap)?;
            offsets.push(layer_offsets);
            maps.push(mmap);
        }
        Ok(Self { maps, offsets })
    }

    /// Extract specific experts from a layer's mmap'd safetensors.
    /// Creates 9 MLX arrays, each with shape [num_experts, d1, d2],
    /// containing only the requested expert slices (copied from mmap).
    pub fn extract_experts(&self, layer: usize, expert_indices: &[i32]) -> ExpertSlice {
        let mmap = &self.maps[layer];
        let layer_offsets = &self.offsets[layer];
        let n = expert_indices.len() as i32;

        let mut arrays = Vec::with_capacity(9);
        for tensor in &layer_offsets.tensors {
            // Build a contiguous buffer with only the requested experts
            let mut buf = Vec::with_capacity(expert_indices.len() * tensor.per_expert_stride);
            for &eidx in expert_indices {
                let abs_start = layer_offsets.data_start
                    + tensor.data_offset
                    + eidx as usize * tensor.per_expert_stride;
                let slice = &mmap[abs_start..abs_start + tensor.per_expert_stride];
                buf.extend_from_slice(slice);
            }
            // Shape: [num_experts, ...expert_shape]
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

        ExpertSlice {
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

    /// Prefetch warm set expert pages into kernel page cache.
    /// Uses madvise(MADV_WILLNEED) to hint the kernel without pinning.
    /// Returns total bytes advised.
    pub fn mlock_warm_set(&self, experts: &[(u32, u32)]) -> usize {
        let page_size: usize = 16384; // Apple Silicon page size
        let mut advised = 0usize;

        for &(layer, expert_idx) in experts {
            let layer = layer as usize;
            let expert_idx = expert_idx as usize;

            if layer >= self.maps.len() {
                continue;
            }
            let mmap = &self.maps[layer];
            let layer_offsets = &self.offsets[layer];

            for tensor in &layer_offsets.tensors {
                let abs_start = layer_offsets.data_start
                    + tensor.data_offset
                    + expert_idx * tensor.per_expert_stride;
                let len = tensor.per_expert_stride;

                let aligned_start = abs_start & !(page_size - 1);
                let aligned_len = (abs_start + len - aligned_start + page_size - 1)
                    & !(page_size - 1);

                unsafe {
                    let ptr = mmap.as_ptr().add(aligned_start);
                    libc::madvise(ptr as *mut _, aligned_len, libc::MADV_WILLNEED);
                    advised += aligned_len;
                }
            }
        }

        advised
    }
}

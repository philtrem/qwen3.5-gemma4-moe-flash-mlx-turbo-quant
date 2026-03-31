use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;

use memmap2::Mmap;
use serde_json::Value;

/// Split a model into resident weights (safetensors) and per-layer expert files.
/// `format` is "ecb" (expert-centric binary) or "safetensors".
pub fn split_model(model_path: &Path, output_path: &Path, format: &str) -> io::Result<()> {
    fs::create_dir_all(output_path)?;
    let resident_dir = output_path.join("resident");
    let expert_dir = output_path.join("experts");
    fs::create_dir_all(&resident_dir)?;
    fs::create_dir_all(&expert_dir)?;

    // Copy config files
    for name in &[
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "chat_template.jinja",
    ] {
        let src = model_path.join(name);
        if src.exists() {
            fs::copy(&src, output_path.join(name))?;
        }
    }

    // Read weight index
    let index_path = model_path.join("model.safetensors.index.json");
    let index_str = fs::read_to_string(&index_path)?;
    let index: Value = serde_json::from_str(&index_str)?;
    let weight_map = index["weight_map"]
        .as_object()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "no weight_map in index"))?;

    // Classify tensors
    let mut resident_tensors: Vec<String> = Vec::new();
    let mut expert_tensors: Vec<String> = Vec::new();
    for key in weight_map.keys() {
        if key.contains("switch_mlp") {
            expert_tensors.push(key.clone());
        } else {
            resident_tensors.push(key.clone());
        }
    }

    eprintln!(
        "Found {} resident tensors, {} expert tensors",
        resident_tensors.len(),
        expert_tensors.len()
    );

    // Open all shard files
    let mut shard_mmaps: HashMap<String, Mmap> = HashMap::new();
    let mut shard_files: Vec<String> = weight_map
        .values()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();
    shard_files.sort();
    shard_files.dedup();

    for shard_name in &shard_files {
        let shard_path = model_path.join(shard_name);
        let file = File::open(&shard_path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        shard_mmaps.insert(shard_name.clone(), mmap);
    }

    // Step 1: Write resident weights as safetensors
    write_resident_weights(&shard_mmaps, weight_map, &resident_tensors, &resident_dir)?;

    // Step 2: Write expert weights in requested format
    match format {
        "ecb" => write_expert_ecb(&shard_mmaps, weight_map, &expert_dir)?,
        "safetensors" => return Err(io::Error::new(io::ErrorKind::InvalidInput,
            "safetensors expert format is no longer supported (inference requires ECB for zero-copy Metal buffers). Use --format ecb (the default).")),
        _ => return Err(io::Error::new(io::ErrorKind::InvalidInput,
            format!("unknown format '{}', expected 'ecb'", format))),
    }

    // Write split metadata
    let meta = serde_json::json!({
        "original_model": model_path.to_str(),
        "resident_dir": "resident",
        "expert_dir": "experts",
        "format": format,
    });
    fs::write(
        output_path.join("split_config.json"),
        serde_json::to_string_pretty(&meta).unwrap(),
    )?;

    eprintln!("Model split complete: {}", output_path.display());
    Ok(())
}

/// Parse a safetensors shard and return (header_json, data_offset).
fn parse_shard(mmap: &[u8]) -> io::Result<(Value, usize)> {
    let header_size = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
    let header: Value = serde_json::from_slice(&mmap[8..8 + header_size])
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
    Ok((header, 8 + header_size))
}

/// Extract raw tensor bytes from a shard mmap.
fn extract_tensor<'a>(mmap: &'a [u8], header: &Value, tensor_name: &str) -> io::Result<&'a [u8]> {
    let info = header.get(tensor_name).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!("tensor {} not in shard", tensor_name),
        )
    })?;
    let offsets = info["data_offsets"]
        .as_array()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "no data_offsets"))?;
    let start = offsets[0].as_u64().unwrap() as usize;
    let end = offsets[1].as_u64().unwrap() as usize;

    let header_size = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
    let data_base = 8 + header_size;

    Ok(&mmap[data_base + start..data_base + end])
}

/// Get tensor shape from shard header.
fn tensor_shape(header: &Value, tensor_name: &str) -> io::Result<Vec<usize>> {
    let info = header
        .get(tensor_name)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, tensor_name.to_string()))?;
    Ok(info["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect())
}

/// Get tensor dtype from shard header.
fn tensor_dtype(header: &Value, tensor_name: &str) -> io::Result<String> {
    let info = header
        .get(tensor_name)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, tensor_name.to_string()))?;
    Ok(info["dtype"].as_str().unwrap_or("F32").to_string())
}

fn parse_all_shard_headers(shard_mmaps: &HashMap<String, Mmap>) -> io::Result<HashMap<String, Value>> {
    let mut headers = HashMap::new();
    for (name, mmap) in shard_mmaps {
        let (header, _) = parse_shard(mmap)?;
        headers.insert(name.clone(), header);
    }
    Ok(headers)
}

fn discover_num_expert_layers(weight_map: &serde_json::Map<String, Value>) -> u32 {
    let mut max_layer = 0u32;
    for key in weight_map.keys() {
        if key.contains("switch_mlp") {
            if let Some(rest) = key.strip_prefix("language_model.model.layers.") {
                if let Some(dot_pos) = rest.find('.') {
                    if let Ok(idx) = rest[..dot_pos].parse::<u32>() {
                        max_layer = max_layer.max(idx);
                    }
                }
            }
        }
    }
    max_layer + 1
}

/// Write resident (non-expert) weights as safetensors.
fn write_resident_weights(
    shard_mmaps: &HashMap<String, Mmap>,
    weight_map: &serde_json::Map<String, Value>,
    resident_tensors: &[String],
    output_dir: &Path,
) -> io::Result<()> {
    let shard_headers = parse_all_shard_headers(shard_mmaps)?;

    let mut tensor_data: Vec<(String, Vec<u8>, String, Vec<usize>)> = Vec::new();

    for tensor_name in resident_tensors {
        let shard_name = weight_map[tensor_name].as_str().unwrap();
        let mmap = &shard_mmaps[shard_name];
        let header = &shard_headers[shard_name];

        let data = extract_tensor(mmap, header, tensor_name)?;
        let shape = tensor_shape(header, tensor_name)?;
        let dtype = tensor_dtype(header, tensor_name)?;

        // Strip "language_model." prefix
        let clean_name = tensor_name
            .strip_prefix("language_model.")
            .unwrap_or(tensor_name)
            .to_string();

        tensor_data.push((clean_name, data.to_vec(), dtype, shape));
    }

    tensor_data.sort_by(|a, b| a.0.cmp(&b.0));
    write_safetensors_file(&tensor_data, &output_dir.join("resident.safetensors"))?;

    // Write index file
    let mut new_weight_map = serde_json::Map::new();
    for (name, _, _, _) in &tensor_data {
        new_weight_map.insert(name.clone(), Value::String("resident.safetensors".to_string()));
    }
    let index = serde_json::json!({
        "metadata": { "format": "mlx" },
        "weight_map": new_weight_map,
    });
    fs::write(
        output_dir.join("model.safetensors.index.json"),
        serde_json::to_string_pretty(&index).unwrap(),
    )?;

    let total_bytes: usize = tensor_data.iter().map(|(_, d, _, _)| d.len()).sum();
    eprintln!(
        "Wrote {} resident tensors ({:.2} GB)",
        tensor_data.len(),
        total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
    );

    Ok(())
}

/// Write per-layer expert safetensors files.
///
/// Each file contains 9 tensors (gate/up/down × weight/scales/biases),
/// each with shape (256, d1, d2) — all experts stacked.
fn write_expert_safetensors(
    shard_mmaps: &HashMap<String, Mmap>,
    weight_map: &serde_json::Map<String, Value>,
    expert_dir: &Path,
) -> io::Result<()> {
    let shard_headers = parse_all_shard_headers(shard_mmaps)?;
    let num_layers = discover_num_expert_layers(weight_map);

    eprintln!("Writing {} layers of expert safetensors...", num_layers);

    for layer_idx in 0..num_layers {
        let prefix = format!(
            "language_model.model.layers.{}.mlp.switch_mlp",
            layer_idx
        );

        // Collect the 9 expert tensors for this layer
        let proj_names = ["gate_proj", "up_proj", "down_proj"];
        let comp_names = ["weight", "scales", "biases"];

        let mut tensors: Vec<(String, Vec<u8>, String, Vec<usize>)> = Vec::new();

        for proj in &proj_names {
            for comp in &comp_names {
                let tensor_name = format!("{}.{}.{}", prefix, proj, comp);
                let shard_name = weight_map
                    .get(&tensor_name)
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::NotFound,
                            format!("tensor {} not in weight map", tensor_name),
                        )
                    })?;
                let mmap = &shard_mmaps[shard_name];
                let header = &shard_headers[shard_name];

                let data = extract_tensor(mmap, header, &tensor_name)?;
                let shape = tensor_shape(header, &tensor_name)?;
                let dtype = tensor_dtype(header, &tensor_name)?;

                // Use short name: "gate_proj.weight", "up_proj.scales", etc.
                let short_name = format!("{}.{}", proj, comp);
                tensors.push((short_name, data.to_vec(), dtype, shape));
            }
        }

        let file_path = expert_dir.join(format!("layer_{:02}_experts.safetensors", layer_idx));
        write_safetensors_file(&tensors, &file_path)?;

        if (layer_idx + 1) % 10 == 0 || layer_idx == num_layers - 1 {
            eprintln!("  Wrote layer {}/{}", layer_idx + 1, num_layers);
        }
    }

    Ok(())
}

/// ECB dtype encoding (matches memory.rs parser).
fn ecb_dtype_code(dtype_str: &str) -> u32 {
    match dtype_str {
        "U8" => 0,
        "U32" => 1,
        "BF16" => 2,
        "F16" => 3,
        "F32" => 4,
        "I32" => 5,
        _ => panic!("unsupported dtype for ECB: {}", dtype_str),
    }
}

/// Write per-layer expert-centric binary (ECB) files.
///
/// Rearranges tensor-centric layout (all experts for tensor 0, then tensor 1, ...)
/// into expert-centric layout (all tensors for expert 0, then expert 1, ...).
/// This enables reading all data for one expert in a single contiguous pread.
fn write_expert_ecb(
    shard_mmaps: &HashMap<String, Mmap>,
    weight_map: &serde_json::Map<String, Value>,
    expert_dir: &Path,
) -> io::Result<()> {
    let shard_headers = parse_all_shard_headers(shard_mmaps)?;
    let num_layers = discover_num_expert_layers(weight_map);

    eprintln!("Writing {} layers of expert ECB files...", num_layers);

    let proj_names = ["gate_proj", "up_proj", "down_proj"];
    let comp_names = ["weight", "scales", "biases"];

    for layer_idx in 0..num_layers {
        let prefix = format!("language_model.model.layers.{}.mlp.switch_mlp", layer_idx);

        // Collect 9 tensors: (data, per_expert_stride, dtype_code, per_expert_shape)
        struct TensorMeta {
            data: Vec<u8>,             // full [256, ...] data, row-major
            per_expert_stride: usize,  // bytes per single expert
            dtype_code: u32,
            expert_shape: Vec<u32>,    // shape excluding expert dim
            num_experts: usize,
        }

        let mut tensor_metas: Vec<TensorMeta> = Vec::with_capacity(9);

        for proj in &proj_names {
            for comp in &comp_names {
                let tensor_name = format!("{}.{}.{}", prefix, proj, comp);
                let shard_name = weight_map.get(&tensor_name).and_then(|v| v.as_str())
                    .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound,
                        format!("tensor {} not in weight map", tensor_name)))?;
                let mmap = &shard_mmaps[shard_name];
                let header = &shard_headers[shard_name];

                let data = extract_tensor(mmap, header, &tensor_name)?;
                let shape = tensor_shape(header, &tensor_name)?;
                let dtype_str = tensor_dtype(header, &tensor_name)?;

                let num_experts = shape[0];
                let expert_shape: Vec<u32> = shape[1..].iter().map(|&s| s as u32).collect();
                let per_expert_stride = data.len() / num_experts;

                tensor_metas.push(TensorMeta {
                    data: data.to_vec(),
                    per_expert_stride,
                    dtype_code: ecb_dtype_code(&dtype_str),
                    expert_shape,
                    num_experts,
                });
            }
        }

        let num_experts = tensor_metas[0].num_experts as u32;
        let per_expert_stride: usize = tensor_metas.iter().map(|t| t.per_expert_stride).sum();
        let num_tensors = tensor_metas.len() as u32;
        let header_size: u32 = 16384;

        // Build ECB header
        let mut header_buf = vec![0u8; header_size as usize];
        let mut pos = 0usize;

        // Magic
        header_buf[pos..pos + 4].copy_from_slice(b"ECB1");
        pos += 4;
        // num_experts
        header_buf[pos..pos + 4].copy_from_slice(&num_experts.to_le_bytes());
        pos += 4;
        // per_expert_stride
        header_buf[pos..pos + 4].copy_from_slice(&(per_expert_stride as u32).to_le_bytes());
        pos += 4;
        // num_tensors
        header_buf[pos..pos + 4].copy_from_slice(&num_tensors.to_le_bytes());
        pos += 4;
        // header_size
        header_buf[pos..pos + 4].copy_from_slice(&header_size.to_le_bytes());
        pos += 4;

        // Tensor descriptors
        for t in &tensor_metas {
            // stride
            header_buf[pos..pos + 4].copy_from_slice(&(t.per_expert_stride as u32).to_le_bytes());
            pos += 4;
            // dtype
            header_buf[pos..pos + 4].copy_from_slice(&t.dtype_code.to_le_bytes());
            pos += 4;
            // ndim
            header_buf[pos..pos + 4].copy_from_slice(&(t.expert_shape.len() as u32).to_le_bytes());
            pos += 4;
            // shape
            for &dim in &t.expert_shape {
                header_buf[pos..pos + 4].copy_from_slice(&dim.to_le_bytes());
                pos += 4;
            }
        }

        // Write file
        let file_path = expert_dir.join(format!("layer_{:02}_experts.ecb", layer_idx));
        let mut file = File::create(&file_path)?;

        // Header (zero-padded to 16384)
        file.write_all(&header_buf)?;

        // Expert-centric data: for each expert, write its slice from each tensor contiguously
        for expert_idx in 0..num_experts as usize {
            for t in &tensor_metas {
                let start = expert_idx * t.per_expert_stride;
                let end = start + t.per_expert_stride;
                file.write_all(&t.data[start..end])?;
            }
        }

        file.sync_all()?;

        if (layer_idx + 1) % 10 == 0 || layer_idx == num_layers - 1 {
            eprintln!("  Wrote layer {}/{} ({:.1} MB)",
                layer_idx + 1, num_layers,
                (header_size as usize + num_experts as usize * per_expert_stride) as f64 / 1e6);
        }
    }

    Ok(())
}

/// Write a safetensors file from tensor data.
fn write_safetensors_file(
    tensors: &[(String, Vec<u8>, String, Vec<usize>)],
    path: &Path,
) -> io::Result<()> {
    let mut header_map = serde_json::Map::new();
    let mut offset = 0u64;

    for (name, data, dtype, shape) in tensors {
        let end = offset + data.len() as u64;
        header_map.insert(
            name.clone(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [offset, end],
            }),
        );
        offset = end;
    }

    header_map.insert(
        "__metadata__".to_string(),
        serde_json::json!({"format": "pt"}),
    );

    let header_json = serde_json::to_string(&Value::Object(header_map))?;
    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;

    let mut file = File::create(path)?;
    file.write_all(&header_len.to_le_bytes())?;
    file.write_all(header_bytes)?;
    for (_, data, _, _) in tensors {
        file.write_all(data)?;
    }
    file.sync_all()?;

    Ok(())
}

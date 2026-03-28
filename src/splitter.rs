use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::Path;

use memmap2::Mmap;
use serde_json::Value;

/// Split a model into resident weights (safetensors) and per-layer expert safetensors.
pub fn split_model(model_path: &Path, output_path: &Path) -> io::Result<()> {
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

    // Step 2: Write expert weights as per-layer safetensors
    write_expert_safetensors(&shard_mmaps, weight_map, &expert_dir)?;

    // Write split metadata
    let meta = serde_json::json!({
        "original_model": model_path.to_str(),
        "resident_dir": "resident",
        "expert_dir": "experts",
        "format": "safetensors",
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

/// Write resident (non-expert) weights as safetensors.
fn write_resident_weights(
    shard_mmaps: &HashMap<String, Mmap>,
    weight_map: &serde_json::Map<String, Value>,
    resident_tensors: &[String],
    output_dir: &Path,
) -> io::Result<()> {
    let mut shard_headers: HashMap<String, Value> = HashMap::new();
    for (name, mmap) in shard_mmaps {
        let (header, _) = parse_shard(mmap)?;
        shard_headers.insert(name.clone(), header);
    }

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
    let mut shard_headers: HashMap<String, Value> = HashMap::new();
    for (name, mmap) in shard_mmaps {
        let (header, _) = parse_shard(mmap)?;
        shard_headers.insert(name.clone(), header);
    }

    // Discover number of layers
    let num_layers = {
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
    };

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

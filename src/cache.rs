use mlx_rs::error::Exception;
use mlx_rs::Array;

// ============================================================
// TurboQuant support: Randomized Hadamard Transform + Lloyd-Max
// ============================================================

/// Generate normalized Hadamard matrix H_n (Sylvester construction).
/// H * H^T = I for the normalized version. n must be a power of 2.
fn generate_hadamard(n: usize) -> Vec<f32> {
    let mut h = vec![0.0f32; n * n];
    h[0] = 1.0;
    let mut size = 1;
    while size < n {
        for i in 0..size {
            for j in 0..size {
                let val = h[i * n + j];
                h[i * n + (j + size)] = val;
                h[(i + size) * n + j] = val;
                h[(i + size) * n + (j + size)] = -val;
            }
        }
        size *= 2;
    }
    let scale = 1.0 / (n as f32).sqrt();
    for val in &mut h {
        *val *= scale;
    }
    h
}

/// Deterministic ±1 sign vector via LCG PRNG.
fn deterministic_signs(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            if (state >> 63) & 1 == 0 {
                1.0
            } else {
                -1.0
            }
        })
        .collect()
}

/// Build Randomized Hadamard Transform matrices.
/// R[i][j] = H[i][j] * signs[j],  R^T[i][j] = H[i][j] * signs[i].
fn build_rht(dim: usize) -> (Array, Array) {
    let h = generate_hadamard(dim);
    let signs = deterministic_signs(dim, 42);
    let mut r = vec![0.0f32; dim * dim];
    let mut rt = vec![0.0f32; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            r[i * dim + j] = h[i * dim + j] * signs[j];
            rt[i * dim + j] = h[i * dim + j] * signs[i];
        }
    }
    (
        Array::from_slice(&r, &[dim as i32, dim as i32]),
        Array::from_slice(&rt, &[dim as i32, dim as i32]),
    )
}

/// Lloyd-Max optimal centroids and decision boundaries for N(0,1).
/// After rotating a unit vector with dim d, each coordinate is ~N(0, 1/d).
/// We scale data by sqrt(d) before quantization so these N(0,1) tables apply.
fn lloyd_max_codebook(bits: u8) -> (&'static [f32], &'static [f32]) {
    match bits {
        2 => (
            &[-1.510, -0.453, 0.453, 1.510],
            &[-0.982, 0.0, 0.982],
        ),
        3 => (
            &[-2.152, -1.344, -0.756, -0.245, 0.245, 0.756, 1.344, 2.152],
            &[-1.748, -1.050, -0.501, 0.0, 0.501, 1.050, 1.748],
        ),
        4 => (
            &[
                -2.733, -2.069, -1.618, -1.256, -0.942, -0.657, -0.388, -0.128,
                 0.128,  0.388,  0.657,  0.942,  1.256,  1.618,  2.069,  2.733,
            ],
            &[
                -2.401, -1.844, -1.437, -1.099, -0.800, -0.522, -0.258, 0.0,
                 0.258,  0.522,  0.800,  1.099,  1.437,  1.844,  2.401,
            ],
        ),
        _ => panic!("TurboQuant supports 2, 3, or 4 bits, got {}", bits),
    }
}

/// TurboQuant quantize: normalize → rotate → scale → boundary search.
/// Input x: [B, H, T, D] in any dtype.
/// Returns (indices: [B, H, T, D] uint8, norms: [B, H, T, 1] f32).
fn turbo_quantize(
    x: &Array,
    boundaries: &Array,
    rotation: &Array,
    sqrt_d: f32,
) -> Result<(Array, Array), Exception> {
    let x = x.as_dtype(mlx_rs::Dtype::Float32)?;

    // L2 norm per vector: [B, H, T, 1]
    let x_sq = &x * &x;
    let sum_sq = mlx_rs::ops::sum_axis(&x_sq, -1, Some(true))?;
    let norm = mlx_rs::ops::sqrt(&sum_sq)?;

    // Normalize (epsilon prevents division by zero)
    let eps = Array::from_f32(1e-8);
    let safe_norm = &norm + &eps;
    let x_unit = &x / &safe_norm;

    // Rotate: [B, H, T, D] @ [D, D]
    let x_rot = mlx_rs::ops::matmul(&x_unit, rotation)?;

    // Scale to ~N(0,1) for codebook lookup
    let scale = Array::from_f32(sqrt_d);
    let x_scaled = &x_rot * &scale;

    // Searchsorted via boundary comparison: sum(x[..., None] >= boundaries, axis=-1)
    let x_exp = mlx_rs::ops::expand_dims(&x_scaled, -1)?; // [B, H, T, D, 1]
    let cmp = x_exp.ge(boundaries)?; // [B, H, T, D, num_boundaries]
    let indices = mlx_rs::ops::sum_axis(&cmp, -1, Some(false))?; // [B, H, T, D]
    let indices = indices.as_dtype(mlx_rs::Dtype::Uint8)?;

    Ok((indices, norm))
}

/// TurboQuant dequantize: lookup → unscale → inverse rotate → scale by norm.
/// Returns [B, H, T, D] in bf16 to keep SDPA in the fast bf16 compute path.
/// (Model is 9-bit quantized — bf16 is already more precision than activations carry.)
fn turbo_dequantize(
    indices: &Array,
    norms: &Array,
    codebook: &Array,
    rotation_t: &Array,
    sqrt_d: f32,
) -> Result<Array, Exception> {
    // Codebook lookup: flatten → take → reshape
    let idx = indices.as_dtype(mlx_rs::Dtype::Int32)?;
    let shape = indices.shape().to_vec();
    let flat = idx.flatten(None, None)?;
    let flat_vals = mlx_rs::ops::indexing::take_axis(codebook, &flat, 0)?;
    let vals = flat_vals.reshape(&shape)?;

    // Undo N(0,1) scaling
    let inv_scale = Array::from_f32(1.0 / sqrt_d);
    let x_rot = &vals * &inv_scale;

    // Inverse rotation: [B, H, T, D] @ [D, D]
    let x_unit = mlx_rs::ops::matmul(&x_rot, rotation_t)?;

    // Scale by original norm, cast to bf16
    let result = &x_unit * norms;
    result.as_dtype(mlx_rs::Dtype::Bfloat16)
}

// ============================================================
// KV Cache
// ============================================================

/// KV cache for full attention layers (10 of 40).
/// Supports both plain bf16 storage and TurboQuant compression.
pub struct KVCache {
    inner: KVCacheInner,
    offset: usize,
}

enum KVCacheInner {
    /// Standard bf16 key/value storage.
    Plain {
        keys: Option<Array>,
        values: Option<Array>,
    },
    /// TurboQuant: rotation + Lloyd-Max codebook quantization.
    Quantized {
        key_indices: Option<Array>,
        key_norms: Option<Array>,
        value_indices: Option<Array>,
        value_norms: Option<Array>,
        codebook: Array,
        boundaries: Array,
        rotation: Array,
        rotation_t: Array,
        sqrt_d: f32,
    },
}

impl KVCache {
    /// Create a plain (unquantized) KV cache.
    pub fn new() -> Self {
        Self {
            inner: KVCacheInner::Plain {
                keys: None,
                values: None,
            },
            offset: 0,
        }
    }

    /// Create a TurboQuant-compressed KV cache.
    pub fn new_quantized(head_dim: usize, bits: u8) -> Self {
        let (centroids, bounds) = lloyd_max_codebook(bits);
        let codebook = Array::from_slice(centroids, &[centroids.len() as i32]);
        let boundaries = Array::from_slice(bounds, &[bounds.len() as i32]);
        let (rotation, rotation_t) = build_rht(head_dim);
        let sqrt_d = (head_dim as f32).sqrt();

        Self {
            inner: KVCacheInner::Quantized {
                key_indices: None,
                key_norms: None,
                value_indices: None,
                value_norms: None,
                codebook,
                boundaries,
                rotation,
                rotation_t,
                sqrt_d,
            },
            offset: 0,
        }
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Store new keys/values and return all cached K/V for SDPA.
    /// For plain cache: concatenate and return bf16.
    /// For quantized: quantize new, concatenate indices/norms, dequantize all.
    pub fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        match &mut self.inner {
            KVCacheInner::Plain {
                keys: cached_keys,
                values: cached_values,
            } => {
                let (k, v) = match (cached_keys.as_ref(), cached_values.as_ref()) {
                    (Some(ck), Some(cv)) => {
                        let k = mlx_rs::ops::concatenate_axis(&[ck, &keys], 2)?;
                        let v = mlx_rs::ops::concatenate_axis(&[cv, &values], 2)?;
                        (k, v)
                    }
                    _ => (keys, values),
                };
                self.offset = k.dim(2) as usize;
                *cached_keys = Some(k.clone());
                *cached_values = Some(v.clone());
                Ok((k, v))
            }

            KVCacheInner::Quantized {
                key_indices,
                key_norms,
                value_indices,
                value_norms,
                codebook,
                boundaries,
                rotation,
                rotation_t,
                sqrt_d,
            } => {
                // Quantize new K/V
                let (new_ki, new_kn) =
                    turbo_quantize(&keys, boundaries, rotation, *sqrt_d)?;
                let (new_vi, new_vn) =
                    turbo_quantize(&values, boundaries, rotation, *sqrt_d)?;

                // Concatenate with cached quantized data
                let (ki, kn) = match (key_indices.as_ref(), key_norms.as_ref()) {
                    (Some(ci), Some(cn)) => (
                        mlx_rs::ops::concatenate_axis(&[ci, &new_ki], 2)?,
                        mlx_rs::ops::concatenate_axis(&[cn, &new_kn], 2)?,
                    ),
                    _ => (new_ki, new_kn),
                };
                let (vi, vn) = match (value_indices.as_ref(), value_norms.as_ref()) {
                    (Some(ci), Some(cn)) => (
                        mlx_rs::ops::concatenate_axis(&[ci, &new_vi], 2)?,
                        mlx_rs::ops::concatenate_axis(&[cn, &new_vn], 2)?,
                    ),
                    _ => (new_vi, new_vn),
                };

                self.offset = ki.dim(2) as usize;
                *key_indices = Some(ki.clone());
                *key_norms = Some(kn.clone());
                *value_indices = Some(vi.clone());
                *value_norms = Some(vn.clone());

                // Dequantize full cache for SDPA (f32 — MLX auto-promotes with bf16 queries)
                let k = turbo_dequantize(&ki, &kn, codebook, rotation_t, *sqrt_d)?;
                let v = turbo_dequantize(&vi, &vn, codebook, rotation_t, *sqrt_d)?;

                Ok((k, v))
            }
        }
    }
}

// ============================================================
// Arrays cache (linear attention) — unchanged
// ============================================================

/// Arrays cache for linear attention layers (GatedDeltaNet, 30 of 40).
pub struct ArraysCache {
    pub items: Vec<Option<Array>>,
}

impl ArraysCache {
    pub fn new(size: usize) -> Self {
        Self {
            items: (0..size).map(|_| None).collect(),
        }
    }

    pub fn get(&self, idx: usize) -> Option<&Array> {
        self.items[idx].as_ref()
    }

    pub fn set(&mut self, idx: usize, value: Array) {
        self.items[idx] = Some(value);
    }
}

// ============================================================
// Unified cache enum
// ============================================================

pub enum Cache {
    KV(KVCache),
    Arrays(ArraysCache),
}

impl Cache {
    pub fn as_kv_mut(&mut self) -> &mut KVCache {
        match self {
            Cache::KV(kv) => kv,
            _ => panic!("expected KVCache"),
        }
    }

    pub fn as_arrays_mut(&mut self) -> &mut ArraysCache {
        match self {
            Cache::Arrays(ac) => ac,
            _ => panic!("expected ArraysCache"),
        }
    }

    pub fn kv_offset(&self) -> usize {
        match self {
            Cache::KV(kv) => kv.offset(),
            _ => 0,
        }
    }
}

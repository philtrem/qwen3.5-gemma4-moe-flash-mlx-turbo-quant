use mlx_rs::error::Exception;
use mlx_rs::Array;

/// KV cache for full attention layers (10 of 40).
pub struct KVCache {
    keys: Option<Array>,
    values: Option<Array>,
    offset: usize,
}

impl KVCache {
    pub fn new() -> Self {
        Self {
            keys: None,
            values: None,
            offset: 0,
        }
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> Result<(Array, Array), Exception> {
        let (k, v) = match (&self.keys, &self.values) {
            (Some(ck), Some(cv)) => {
                let k = mlx_rs::ops::concatenate_axis(&[ck, &keys], 2)?;
                let v = mlx_rs::ops::concatenate_axis(&[cv, &values], 2)?;
                (k, v)
            }
            _ => (keys, values),
        };
        self.offset = k.dim(2) as usize;
        self.keys = Some(k.clone());
        self.values = Some(v.clone());
        Ok((k, v))
    }
}

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

/// Unified cache enum.
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

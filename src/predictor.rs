use std::collections::HashSet;

use crate::memory::ExpertMemoryManager;

/// Simple predictor: assumes next token uses the same experts as the previous token.
/// Issues F_RDADVISE between tokens to pre-warm the page cache.
pub struct ExpertPredictor {
    num_layers: usize,
    /// Previous token's expert selections per layer
    prev: Vec<Vec<i32>>,
    /// Current token's selections (accumulating during forward pass)
    current: Vec<Vec<i32>>,
    /// Accuracy tracking
    total_actual: usize,
    total_hits: usize,
}

impl ExpertPredictor {
    pub fn new(num_layers: usize) -> Self {
        Self {
            num_layers,
            prev: vec![Vec::new(); num_layers],
            current: vec![Vec::new(); num_layers],
            total_actual: 0,
            total_hits: 0,
        }
    }

    /// Record one layer's routing decision. Called from moe.rs after routing.
    pub fn record(&mut self, layer: usize, expert_indices: &[i32]) {
        self.current[layer] = expert_indices.to_vec();
    }

    /// End of token: measure prediction accuracy, then swap current → prev.
    pub fn end_token(&mut self) {
        // Measure: how many of current token's experts were predicted (= in prev)?
        if !self.prev[0].is_empty() {
            for layer in 0..self.num_layers {
                let prev_set: HashSet<i32> = self.prev[layer].iter().copied().collect();
                for &e in &self.current[layer] {
                    self.total_actual += 1;
                    if prev_set.contains(&e) {
                        self.total_hits += 1;
                    }
                }
            }
        }

        std::mem::swap(&mut self.prev, &mut self.current);
        for layer in &mut self.current {
            layer.clear();
        }
    }

    /// Issue F_RDADVISE for previous token's experts across all layers.
    /// Call between tokens when GPU is busy with sampling.
    pub fn prefetch_next_token(&self, mem: &ExpertMemoryManager) {
        if self.prev[0].is_empty() {
            return;
        }
        for layer in 0..self.num_layers {
            mem.prefetch_experts(layer, &self.prev[layer]);
        }
    }

    /// Return (actual, hits, hit_rate) and reset.
    pub fn take_stats(&mut self) -> (usize, usize, f64) {
        let a = self.total_actual;
        let h = self.total_hits;
        let rate = if a > 0 { h as f64 / a as f64 } else { 0.0 };
        self.total_actual = 0;
        self.total_hits = 0;
        (a, h, rate)
    }
}

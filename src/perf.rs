use std::cell::Cell;
use std::time::Duration;

/// Per-phase timing accumulator for decode performance analysis.
/// All values in nanoseconds. Single-threaded (Cell, not Atomic).
pub struct PerfStats {
    // Eval barriers — where GPU actually syncs
    pub gdn_proj_eval: Cell<u64>,     // eval([q,k,v,beta,g]) — prefill only (0 during decode)
    pub moe_routing_eval: Cell<u64>,  // eval(flat_idx) — routing + attn tail (40/tok)
    pub moe_sort_eval: Cell<u64>,     // eval([x_sorted,idx_sorted]) — argsort boundary (40/tok)
    pub layer_eval: Cell<u64>,        // eval(h) total wall time (40/tok)
    pub eval_wait: Cell<u64>,         // eval(h) after async_eval — page fault time (40/tok)

    // CPU work between evals
    pub extract_experts: Cell<u64>,   // pread I/O + from_raw_data (40/tok)
    pub routing_cpu: Cell<u64>,       // unique/sort/dedup/HashMap/remap + prefetch (40/tok)
    pub sync_mlock: Cell<u64>,        // mlock blocking time before eval (30/tok, MoE only)
}

impl PerfStats {
    pub fn new() -> Self {
        Self {
            gdn_proj_eval: Cell::new(0),
            moe_routing_eval: Cell::new(0),
            moe_sort_eval: Cell::new(0),
            layer_eval: Cell::new(0),
            eval_wait: Cell::new(0),
            extract_experts: Cell::new(0),
            routing_cpu: Cell::new(0),
            sync_mlock: Cell::new(0),
        }
    }

    pub fn acc(&self, field: &Cell<u64>, elapsed: Duration) {
        field.set(field.get() + elapsed.as_nanos() as u64);
    }

    pub fn reset(&self) {
        self.gdn_proj_eval.set(0);
        self.moe_routing_eval.set(0);
        self.moe_sort_eval.set(0);
        self.layer_eval.set(0);
        self.eval_wait.set(0);
        self.extract_experts.set(0);
        self.routing_cpu.set(0);
        self.sync_mlock.set(0);
    }

    pub fn report(&self, num_tokens: usize) {
        let ms = |ns: u64| ns as f64 / 1_000_000.0;
        let per_tok = |ns: u64| if num_tokens > 0 { ms(ns) / num_tokens as f64 } else { 0.0 };

        let evals_total = self.gdn_proj_eval.get()
            + self.moe_routing_eval.get()
            + self.moe_sort_eval.get()
            + self.layer_eval.get();
        let cpu_total = self.extract_experts.get() + self.routing_cpu.get() + self.sync_mlock.get();
        let total = evals_total + cpu_total;

        let pct = |ns: u64| if total > 0 { ns as f64 / total as f64 * 100.0 } else { 0.0 };

        eprintln!("\n=== Perf Breakdown ({} decode tokens) ===", num_tokens);
        eprintln!("Phase                    Total ms   ms/tok    %");
        eprintln!("─────────────────────────────────────────────────");
        if self.gdn_proj_eval.get() > 0 {
            eprintln!("GDN proj eval:         {:>8.1}   {:>6.1}   {:>4.1}%",
                ms(self.gdn_proj_eval.get()), per_tok(self.gdn_proj_eval.get()), pct(self.gdn_proj_eval.get()));
        }
        eprintln!("MoE routing eval (×40):{:>8.1}   {:>6.1}   {:>4.1}%",
            ms(self.moe_routing_eval.get()), per_tok(self.moe_routing_eval.get()), pct(self.moe_routing_eval.get()));
        eprintln!("MoE sort eval (×40):   {:>8.1}   {:>6.1}   {:>4.1}%",
            ms(self.moe_sort_eval.get()), per_tok(self.moe_sort_eval.get()), pct(self.moe_sort_eval.get()));
        eprintln!("Layer eval (×40):      {:>8.1}   {:>6.1}   {:>4.1}%",
            ms(self.layer_eval.get()), per_tok(self.layer_eval.get()), pct(self.layer_eval.get()));
        if self.eval_wait.get() > 0 {
            eprintln!("  └ GPU wait:          {:>8.1}   {:>6.1}   {:>4.1}%",
                ms(self.eval_wait.get()), per_tok(self.eval_wait.get()), pct(self.eval_wait.get()));
        }
        eprintln!("─────────────────────────────────────────────────");
        eprintln!("  Eval subtotal:       {:>8.1}   {:>6.1}   {:>4.1}%",
            ms(evals_total), per_tok(evals_total), pct(evals_total));
        eprintln!("─────────────────────────────────────────────────");
        eprintln!("Extract experts (×40): {:>8.1}   {:>6.1}   {:>4.1}%",
            ms(self.extract_experts.get()), per_tok(self.extract_experts.get()), pct(self.extract_experts.get()));
        eprintln!("Routing CPU (×40):     {:>8.1}   {:>6.1}   {:>4.1}%",
            ms(self.routing_cpu.get()), per_tok(self.routing_cpu.get()), pct(self.routing_cpu.get()));
        if self.sync_mlock.get() > 0 {
            eprintln!("Sync mlock (×30):      {:>8.1}   {:>6.1}   {:>4.1}%",
                ms(self.sync_mlock.get()), per_tok(self.sync_mlock.get()), pct(self.sync_mlock.get()));
        }
        eprintln!("─────────────────────────────────────────────────");
        eprintln!("  CPU subtotal:        {:>8.1}   {:>6.1}   {:>4.1}%",
            ms(cpu_total), per_tok(cpu_total), pct(cpu_total));
        eprintln!("─────────────────────────────────────────────────");
        eprintln!("  ACCOUNTED TOTAL:     {:>8.1}   {:>6.1}",
            ms(total), per_tok(total));
        eprintln!("  Implied tok/s:       {:>8.1}", if per_tok(total) > 0.0 { 1000.0 / per_tok(total) } else { 0.0 });
    }
}

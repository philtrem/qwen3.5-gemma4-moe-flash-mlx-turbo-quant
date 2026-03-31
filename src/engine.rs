use std::cell::RefCell;
use std::io::Write;
use std::time::Instant;

use mlx_rs::error::Exception;
use mlx_rs::Array;

use crate::memory::ExpertMemoryManager;
use crate::model::Model;
use crate::model::moe::TransitionProfiler;
use crate::perf::PerfStats;
use crate::tokenizer::QwenTokenizer;

pub fn generate(
    model: &mut Model,
    tokenizer: &QwenTokenizer,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    mem: &ExpertMemoryManager,
    kv_quant_bits: Option<u8>,
    speculate: bool,
) -> anyhow::Result<String> {
    let perf = PerfStats::new();
    let tp = RefCell::new(TransitionProfiler::new(40));
    let input_ids = tokenizer.encode(prompt)?;
    let mut cache = model.make_cache(kv_quant_bits);

    // Prefill
    eprintln!("Prefilling {} tokens...", input_ids.len());
    let t0 = Instant::now();
    let input = Array::from_slice(
        &input_ids.iter().map(|&x| x as i32).collect::<Vec<_>>(),
        &[1, input_ids.len() as i32],
    );
    let logits = model.forward(&input, &mut cache, mem, &perf, false, None)?;
    mlx_rs::transforms::eval(std::iter::once(&logits))?;
    let prefill_time = t0.elapsed();
    eprintln!(
        "Prefill: {:.2}s ({:.1} tok/s)",
        prefill_time.as_secs_f64(),
        input_ids.len() as f64 / prefill_time.as_secs_f64()
    );

    // Sample from last position
    let seq_len = logits.dim(1);
    let last_idx = Array::from_slice(&[seq_len - 1], &[1]);
    let last_logits = mlx_rs::ops::indexing::take_axis(&logits, &last_idx, 1)?;
    let last_logits = mlx_rs::ops::squeeze_axes(&last_logits, &[1])?;
    let mut next_token = sample(&last_logits, temperature, top_p)?;
    mlx_rs::transforms::eval(std::iter::once(&next_token))?;

    let mut generated: Vec<u32> = vec![next_token.item::<i32>() as u32];
    let mut stdout = std::io::stdout();

    let text = tokenizer.decode(&generated)?;
    print!("{}", text);
    stdout.flush().ok();

    // Reset perf stats and cache stats for decode-only measurement
    perf.reset();
    mem.reset_cache_stats();
    let t_start = Instant::now();
    let mut t_interval = Instant::now();
    let mut tokens_generated = 0usize;
    let mut prev_text_len = text.len();

    for _ in 1..max_tokens {
        let tok_id = *generated.last().unwrap();
        if tokenizer.is_eos(tok_id) {
            generated.pop();
            break;
        }

        let input = Array::from_slice(&[tok_id as i32], &[1, 1]);
        let logits = model.forward(&input, &mut cache, mem, &perf, speculate, Some(&tp))?;
        let logits = mlx_rs::ops::squeeze_axes(&logits, &[1])?;
        next_token = sample(&logits, temperature, top_p)?;
        mlx_rs::transforms::eval(std::iter::once(&next_token))?;

        tp.borrow_mut().end_token();

        let new_tok = next_token.item::<i32>() as u32;
        generated.push(new_tok);
        tokens_generated += 1;

        // Stream
        let full_text = tokenizer.decode(&generated)?;
        if full_text.len() > prev_text_len {
            print!("{}", &full_text[prev_text_len..]);
            stdout.flush().ok();
            prev_text_len = full_text.len();
        }

        if tokens_generated % 10 == 0 {
            let elapsed = t_start.elapsed().as_secs_f64();
            let interval_elapsed = t_interval.elapsed().as_secs_f64();
            let interval_rate = 10.0 / interval_elapsed;
            eprint!(
                "\r  {} tokens, {:.1} tok/s (last 10: {:.1} tok/s)",
                tokens_generated,
                tokens_generated as f64 / elapsed,
                interval_rate,
            );
            t_interval = Instant::now();
        }
    }

    println!();
    let elapsed = t_start.elapsed().as_secs_f64();
    eprintln!(
        "\nGeneration: {} tokens in {:.2}s ({:.1} tok/s)",
        tokens_generated, elapsed,
        tokens_generated as f64 / elapsed
    );

    let (hits, misses, rate) = mem.take_hit_stats();
    if hits + misses > 0 {
        eprintln!(
            "Warm set hit rate: {:.1}% ({}/{} expert loads)",
            rate * 100.0, hits, hits + misses
        );
    }

    let (ch, cm, cr) = mem.take_cache_stats();
    if ch + cm > 0 {
        eprintln!(
            "Expert cache hit rate: {:.1}% ({}/{} lookups, cache size {})",
            cr * 100.0, ch, ch + cm,
            mem.cache_size(),
        );
    }

    perf.report(tokens_generated);
    tp.borrow().report();

    Ok(tokenizer.decode(&generated)?)
}

fn sample(logits: &Array, temperature: f32, top_p: f32) -> Result<Array, Exception> {
    if temperature == 0.0 {
        return mlx_rs::ops::indexing::argmax_axis(logits, -1, None);
    }

    let logits = logits / temperature;

    let logits = if top_p < 1.0 {
        let neg_logits = -&logits;
        let sorted_indices = mlx_rs::ops::argsort_axis(&neg_logits, -1)?;
        let sorted_logits = mlx_rs::ops::indexing::take_along_axis(&logits, &sorted_indices, Some(-1))?;
        let probs = mlx_rs::ops::softmax_axis(&sorted_logits, -1, None)?;
        let cumulative = mlx_rs::ops::cumsum(&probs, Some(-1), None, None)?;
        let diff = &cumulative - &probs;
        let threshold = Array::from_f32(top_p);
        let mask = diff.gt(&threshold)?;
        let neg_inf = Array::from_f32(f32::NEG_INFINITY);
        let filtered = mlx_rs::ops::r#where(&mask, &neg_inf, &sorted_logits)?;
        let inv_indices = mlx_rs::ops::argsort_axis(&sorted_indices, -1)?;
        mlx_rs::ops::indexing::take_along_axis(&filtered, &inv_indices, Some(-1))?
    } else {
        logits
    };

    let probs = mlx_rs::ops::softmax_axis(&logits, -1, None)?;
    let log_probs = mlx_rs::ops::log(&(&probs + 1e-10f32))?;
    mlx_rs::random::categorical(&log_probs, Some(-1), None::<mlx_rs::random::ShapeOrCount>, None::<&Array>)
}

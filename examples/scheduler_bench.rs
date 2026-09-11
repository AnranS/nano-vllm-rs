//! Model-free scheduler diagnostics. This does NOT benchmark LLM inference.
//! No weights are loaded and no GPU operations are executed.

use std::{env, fs, time::Instant};

use anyhow::{Context, Result, ensure};
use nano_vllm_rs::runtime::{Batch, LlmEngine, ModelRunner, Request, SchedulerConfig};
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize)]
struct Workload {
    name: String,
    #[serde(default)]
    seed: u64,
    requests: Vec<Request>,
}

#[derive(Default)]
struct CountingRunner {
    calls: usize,
    prefill_calls: usize,
    decode_calls: usize,
}

impl ModelRunner for CountingRunner {
    fn run(&mut self, batch: &Batch) -> Result<Vec<u32>> {
        ensure!(
            self.calls < 100_000,
            "model-free scheduler diagnostic exceeded 100000 calls; possible livelock"
        );
        self.calls += 1;
        if batch.is_decode {
            self.decode_calls += 1;
        } else {
            self.prefill_calls += 1;
        }
        Ok(vec![123; batch.last_token_indices.len()])
    }
}

fn main() -> Result<()> {
    let path = env::args().nth(1).context(
        "usage: cargo run --release --example scheduler_bench -- <workload.json>\n\
         MODEL-FREE diagnostics only; no model or GPU inference",
    )?;
    let workload: Workload = serde_json::from_slice(
        &fs::read(&path).with_context(|| format!("reading workload {path}"))?,
    )
    .context("parsing token workload")?;
    ensure!(!workload.requests.is_empty(), "workload is empty");
    let mut engine = LlmEngine::new(SchedulerConfig::default(), CountingRunner::default())?;
    eprintln!("MODEL-FREE scheduler diagnostic: constant tokens, no GPU inference");
    let started = Instant::now();
    let result = engine.generate(&workload.requests, workload.seed)?;
    let cpu_wall_seconds = started.elapsed().as_secs_f64();
    let runner = engine.runner_mut();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "diagnostic": "model-free scheduler; NOT LLM inference benchmark",
            "workload": workload.name,
            "workload_path": path,
            "cpu_wall_seconds": cpu_wall_seconds,
            "runner_calls": runner.calls,
            "runner_prefill_calls": runner.prefill_calls,
            "runner_decode_calls": runner.decode_calls,
            "stats": {
                "input_tokens": result.stats.input_tokens,
                "output_tokens": result.stats.output_tokens,
                "prefill_tokens": result.stats.prefill_tokens,
                "prefill_steps": result.stats.prefill_steps,
                "decode_steps": result.stats.decode_steps,
                "decode_tokens": result.stats.decode_tokens,
                "preemptions": result.stats.preemptions,
                "cached_tokens": result.stats.cached_tokens,
                "cache_hits": result.stats.cache_hits,
                "peak_kv_blocks": result.stats.peak_kv_blocks,
            },
        }))?
    );
    Ok(())
}

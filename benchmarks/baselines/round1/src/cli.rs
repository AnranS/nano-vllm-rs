use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use serde::Serialize;
use std::path::PathBuf;

#[derive(Parser)]
#[command(version, about = "Single-GPU Qwen3 inference in Rust + CUDA")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate real text using local Qwen3 safetensors weights.
    Generate(GenerateArgs),
    /// Benchmark a saved token workload; excludes model loading and warm-up.
    Bench(BenchArgs),
    /// Export final-position logits for numerical validation.
    Logits(LogitsArgs),
    /// Run the original CPU-only scheduler demonstration (synthetic tokens).
    Demo,
}

#[derive(Args, Clone, Serialize)]
struct ModelArgs {
    /// Local directory containing config.json, tokenizer.json and safetensors.
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value_t = 0)]
    device: i32,
    #[arg(long, default_value_t = 256)]
    block_size: usize,
    #[arg(long, default_value_t = 128)]
    num_blocks: usize,
    #[arg(long, default_value_t = 32)]
    max_sequences: usize,
    #[arg(long, default_value_t = 4096)]
    max_model_len: usize,
    #[arg(long, default_value_t = 1024)]
    max_batch_tokens: usize,
    /// Disable CUDA graph replay.
    #[arg(long)]
    eager: bool,
    #[arg(long)]
    no_prefix_cache: bool,
}

#[derive(Args)]
struct GenerateArgs {
    #[command(flatten)]
    common: ModelArgs,
    /// Repeat --prompt to submit multiple requests together.
    #[arg(long, required = true)]
    prompt: Vec<String>,
    #[arg(long, default_value_t = 128)]
    max_tokens: usize,
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long)]
    ignore_eos: bool,
    /// Wrap user text in Qwen3 chat format with thinking disabled.
    #[arg(long)]
    chat: bool,
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Args)]
struct BenchArgs {
    #[command(flatten)]
    common: ModelArgs,
    #[arg(long)]
    workload: PathBuf,
    #[arg(long, default_value_t = 3)]
    repetitions: usize,
    #[arg(long, default_value_t = 1)]
    warmups: usize,
    /// Optional separate warm-up workload; recorded in the benchmark report.
    #[arg(long)]
    warmup_workload: Option<PathBuf>,
    /// Preserve prefix entries between runs; cold cache is the default.
    #[arg(long)]
    warm_prefix_cache: bool,
    /// Mark measured runs with CUDA profiler start/stop (for Nsight capture).
    #[arg(long)]
    profile_cuda: bool,
    #[arg(long)]
    output: PathBuf,
}

#[derive(Args)]
struct LogitsArgs {
    #[command(flatten)]
    common: ModelArgs,
    /// JSON array of integer token IDs.
    #[arg(long)]
    tokens: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

pub fn run() -> Result<()> {
    let args = Cli::parse();
    match args.command {
        Command::Demo => demo(),
        #[cfg(feature = "cuda")]
        Command::Generate(args) => native::generate(args),
        #[cfg(feature = "cuda")]
        Command::Bench(args) => native::bench(args),
        #[cfg(feature = "cuda")]
        Command::Logits(args) => native::logits(args),
        #[cfg(not(feature = "cuda"))]
        _ => anyhow::bail!("this binary was built without CUDA; rebuild with the default features"),
    }
}

fn demo() -> Result<()> {
    use nano_vllm_rs::{Config, Engine, MockBackend, SamplingParams};
    println!("MOCK DEMO: synthetic token IDs, no model weights or GPU execution.");
    let mut engine = Engine::new(
        Config {
            block_size: 4,
            num_kv_blocks: 4,
            max_running_requests: 2,
        },
        MockBackend,
    )?;
    for prompt in [vec![10, 11], vec![20, 21], vec![30, 31]] {
        engine.add_request(prompt, SamplingParams { max_tokens: 4 })?;
    }
    while !engine.is_finished() {
        for event in engine.step()? {
            println!("{event:?}");
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
mod native {
    use super::*;
    use anyhow::{Context, ensure};
    use nano_vllm_rs::{
        model::{ModelOptions, QwenModel},
        runtime::{LlmEngine, Request, SchedulerConfig},
    };
    use serde::Deserialize;
    use serde_json::{Value, json};
    use std::{
        fs,
        path::Path,
        time::{Instant, SystemTime, UNIX_EPOCH},
    };
    use tokenizers::Tokenizer;

    unsafe extern "C" {
        fn cudaProfilerStart() -> i32;
        fn cudaProfilerStop() -> i32;
    }

    struct ProfilerRange(bool);

    impl ProfilerRange {
        fn start(enabled: bool) -> Result<Self> {
            if enabled {
                // CUDA owns the profiler state; this API takes no pointers.
                ensure!(
                    unsafe { cudaProfilerStart() } == 0,
                    "CUDA profiler start failed"
                );
            }
            Ok(Self(enabled))
        }
    }

    impl Drop for ProfilerRange {
        fn drop(&mut self) {
            if self.0 {
                // End the capture on errors as well as successful completion.
                unsafe { cudaProfilerStop() };
            }
        }
    }

    #[derive(Deserialize)]
    struct Workload {
        name: String,
        #[serde(default)]
        seed: u64,
        requests: Vec<Request>,
    }

    fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, serde_json::to_vec_pretty(value)?)
            .with_context(|| format!("writing {}", path.display()))
    }

    fn load_engine(args: &ModelArgs) -> Result<(LlmEngine<QwenModel>, Value, f64)> {
        let started = Instant::now();
        let model = QwenModel::load(
            &args.model,
            ModelOptions {
                device: args.device,
                block_size: args.block_size,
                num_blocks: args.num_blocks,
                max_batch_tokens: args.max_batch_tokens,
                max_sequences: args.max_sequences,
                max_model_len: args.max_model_len,
                cuda_graphs: !args.eager,
            },
        )?;
        let eos_token_id = model
            .config
            .eos_token_id
            .as_u64()
            .or_else(|| {
                model
                    .config
                    .eos_token_id
                    .as_array()
                    .and_then(|a| a.first())
                    .and_then(Value::as_u64)
            })
            .map(|id| id as u32);
        let memory = serde_json::to_value(model.memory_usage())?;
        let engine = LlmEngine::new(
            SchedulerConfig {
                block_size: args.block_size,
                num_blocks: args.num_blocks,
                max_sequences: args.max_sequences,
                max_model_len: args.max_model_len,
                max_batch_tokens: args.max_batch_tokens,
                prefix_caching: !args.no_prefix_cache,
                eos_token_id,
            },
            model,
        )?;
        let elapsed = started.elapsed().as_secs_f64();
        eprintln!("Loaded {} in {elapsed:.3}s", args.model.display());
        Ok((engine, memory, elapsed))
    }

    fn validate_tokens(engine: &mut LlmEngine<QwenModel>, requests: &[Request]) -> Result<()> {
        let vocab = engine.runner_mut().config.vocab_size;
        for request in requests {
            ensure!(
                request
                    .prompt_token_ids
                    .iter()
                    .all(|&id| (id as usize) < vocab),
                "request {} contains a token ID outside vocabulary size {vocab}",
                request.id
            );
        }
        Ok(())
    }

    pub(super) fn generate(args: GenerateArgs) -> Result<()> {
        let tokenizer = Tokenizer::from_file(args.common.model.join("tokenizer.json"))
            .map_err(|error| anyhow::anyhow!("loading tokenizer: {error}"))?;
        let mut requests = Vec::new();
        for (index, prompt) in args.prompt.iter().enumerate() {
            let text = if args.chat {
                format!(
                    "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                )
            } else {
                prompt.clone()
            };
            let encoding = tokenizer
                .encode(text, false)
                .map_err(|error| anyhow::anyhow!("tokenizing prompt: {error}"))?;
            requests.push(Request {
                id: index as u64,
                prompt_token_ids: encoding.get_ids().to_vec(),
                max_tokens: args.max_tokens,
                temperature: args.temperature,
                ignore_eos: args.ignore_eos,
            });
        }
        let (mut engine, memory, load_seconds) = load_engine(&args.common)?;
        validate_tokens(&mut engine, &requests)?;
        let result = engine.generate(&requests, args.seed)?;
        let mut value = serde_json::to_value(&result)?;
        for entry in value["outputs"]
            .as_array_mut()
            .context("missing generation outputs")?
        {
            let tokens: Vec<u32> = serde_json::from_value(entry["token_ids"].clone())?;
            let text = tokenizer
                .decode(&tokens, true)
                .map_err(|error| anyhow::anyhow!("decoding output: {error}"))?;
            println!("{text}");
            entry["text"] = json!(text);
        }
        value["model_load_seconds"] = json!(load_seconds);
        value["model_memory"] = memory;
        eprintln!("{}", serde_json::to_string(&value["stats"])?);
        if let Some(path) = args.output {
            write_json(&path, &value)?;
        }
        Ok(())
    }

    pub(super) fn bench(args: BenchArgs) -> Result<()> {
        ensure!(
            args.repetitions > 0,
            "repetitions must be greater than zero"
        );
        let workload: Workload = serde_json::from_slice(&fs::read(&args.workload)?)?;
        ensure!(
            !workload.requests.is_empty(),
            "workload must contain at least one request"
        );
        let (mut engine, memory, load_seconds) = load_engine(&args.common)?;
        validate_tokens(&mut engine, &workload.requests)?;
        let warmup: Option<Workload> = args
            .warmup_workload
            .as_ref()
            .map(|path| -> Result<Workload> { Ok(serde_json::from_slice(&fs::read(path)?)?) })
            .transpose()?;
        let warmup = warmup.as_ref().unwrap_or(&workload);
        validate_tokens(&mut engine, &warmup.requests)?;
        for index in 0..args.warmups {
            if !args.warm_prefix_cache {
                engine.clear_prefix_cache()?;
            }
            eprintln!("Warm-up {}/{}", index + 1, args.warmups);
            engine.generate(&warmup.requests, warmup.seed)?;
        }
        let mut runs = Vec::new();
        let profile = ProfilerRange::start(args.profile_cuda)?;
        for index in 0..args.repetitions {
            if !args.warm_prefix_cache {
                engine.clear_prefix_cache()?;
            }
            eprintln!("Measured run {}/{}", index + 1, args.repetitions);
            let run = engine.generate(&workload.requests, workload.seed)?;
            let value = serde_json::to_value(run)?;
            eprintln!("{}", serde_json::to_string(&value["stats"])?);
            runs.push(value);
        }
        drop(profile);
        let result = json!({
            "engine": "nano-vllm-rs",
            "version": env!("CARGO_PKG_VERSION"),
            "timestamp_unix": SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
            "workload": workload.name,
            "workload_path": args.workload,
            "seed": workload.seed,
            "settings": args.common,
            "precision": "bfloat16",
            "warmups": args.warmups,
            "warmup_workload": args.warmup_workload,
            "profile_cuda": args.profile_cuda,
            "prefix_cache_between_runs": if args.warm_prefix_cache { "warm" } else { "cold" },
            "model_load_seconds": load_seconds,
            "model_memory": memory,
            "runs": runs,
        });
        write_json(&args.output, &result)?;
        println!("Benchmark saved to {}", args.output.display());
        Ok(())
    }

    pub(super) fn logits(args: LogitsArgs) -> Result<()> {
        let tokens: Vec<u32> = serde_json::from_slice(&fs::read(&args.tokens)?)?;
        let request = Request {
            id: 0,
            prompt_token_ids: tokens.clone(),
            max_tokens: 1,
            temperature: 0.0,
            ignore_eos: true,
        };
        let (mut engine, _, _) = load_engine(&args.common)?;
        validate_tokens(&mut engine, std::slice::from_ref(&request))?;
        let result = engine.generate(&[request], 0)?;
        let all_logits = engine.runner_mut().last_logits()?;
        ensure!(all_logits.len() == 1, "expected one logits row");
        write_json(
            &args.output,
            &json!({
                "engine": "nano-vllm-rs", "precision": "bfloat16",
                "prompt_token_ids": tokens, "generation": result,
                "logits": all_logits[0],
            }),
        )?;
        println!("Logits saved to {}", args.output.display());
        Ok(())
    }
}

"""Benchmark the local nano-vLLM reference without changing its source files.

Only KV allocation is overridden to match the Rust block budget exactly.
Sampling, scheduling, model layers and attention use the installed reference.
Tokenization, detokenization, startup and warm-up are outside measured runs.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=Path("/root/nano-vllm"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-blocks", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--max-sequences", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-batch-tokens", type=int, default=1024)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--warmup-workload", type=Path)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--warm-prefix-cache", action="store_true")
    parser.add_argument("--profile-cuda", action="store_true")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    sys.path.insert(0, str(args.reference))
    import torch
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.model_runner import ModelRunner
    from nanovllm.engine.scheduler import Scheduler
    from nanovllm.engine.sequence import Sequence

    def allocate_fixed(runner):
        config, hf = runner.config, runner.config.hf_config
        config.num_kvcache_blocks = args.num_blocks
        heads = hf.num_key_value_heads // runner.world_size
        dim = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
        runner.kv_cache = torch.empty(2, hf.num_hidden_layers, args.num_blocks,
                                     config.kvcache_block_size, heads, dim)
        layer = 0
        for module in runner.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = runner.kv_cache[0, layer]
                module.v_cache = runner.kv_cache[1, layer]
                layer += 1
        assert layer == hf.num_hidden_layers

    ModelRunner.allocate_kv_cache = allocate_fixed
    workload = json.loads(args.workload.read_text())
    started = time.perf_counter()
    llm = LLM(str(args.model), enforce_eager=args.eager,
              max_model_len=args.max_model_len, max_num_seqs=args.max_sequences,
              max_num_batched_tokens=args.max_batch_tokens,
              kvcache_block_size=args.block_size, tensor_parallel_size=1)
    load_seconds = time.perf_counter() - started

    def run(selected=workload):
        if not args.warm_prefix_cache:
            llm.scheduler = Scheduler(llm.model_runner.config)
        torch.manual_seed(selected.get("seed", 0))
        torch.cuda.synchronize()
        started = time.perf_counter()
        seqs_by_id = {}
        data = {}
        for request in selected["requests"]:
            seq = Sequence(request["prompt_token_ids"], SamplingParams(
                temperature=request.get("temperature", 0.0),
                ignore_eos=request.get("ignore_eos", True), max_tokens=request["max_tokens"]))
            llm.scheduler.add(seq)
            seqs_by_id[seq.seq_id] = seq
            data[seq.seq_id] = dict(request_id=request["id"], first=None, last=None, gaps=[])
        prefill_tokens = 0
        decode_steps = 0
        while not llm.is_finished():
            seqs, is_prefill = llm.scheduler.schedule()
            previous = {seq.seq_id: seq.num_completion_tokens for seq in seqs}
            if is_prefill:
                prefill_tokens += sum(seq.num_scheduled_tokens for seq in seqs)
            else:
                decode_steps += 1
            token_ids = llm.model_runner.call("run", seqs, is_prefill)
            llm.scheduler.postprocess(seqs, token_ids, is_prefill)
            now = time.perf_counter()
            for seq in seqs:
                if seq.num_completion_tokens == previous[seq.seq_id]:
                    continue
                record = data[seq.seq_id]
                if record["first"] is None:
                    record["first"] = now
                if record["last"] is not None:
                    record["gaps"].append((now - record["last"]) * 1000)
                record["last"] = now
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        outputs = []
        all_gaps = []
        for seq_id, record in data.items():
            seq = seqs_by_id[seq_id]
            all_gaps.extend(record["gaps"])
            outputs.append(dict(request_id=record["request_id"],
                token_ids=seq.completion_token_ids,
                ttft_ms=(record["first"] - started) * 1000,
                latency_ms=(record["last"] - started) * 1000,
                mean_tpot_ms=statistics.mean(record["gaps"]) if record["gaps"] else None))
        total_output = sum(len(o["token_ids"]) for o in outputs)
        return dict(outputs=outputs, stats=dict(elapsed_seconds=elapsed,
            input_tokens=sum(len(r["prompt_token_ids"]) for r in selected["requests"]),
            output_tokens=total_output, output_tokens_per_second=total_output / elapsed,
            prefill_tokens=prefill_tokens, decode_steps=decode_steps,
            ttft_p50_ms=percentile([o["ttft_ms"] for o in outputs], 0.5),
            ttft_p95_ms=percentile([o["ttft_ms"] for o in outputs], 0.95),
            inter_token_p50_ms=percentile(all_gaps, 0.5),
            inter_token_p95_ms=percentile(all_gaps, 0.95)))

    for i in range(args.warmups):
        print(f"Python warm-up {i + 1}/{args.warmups}", flush=True)
        run(json.loads(args.warmup_workload.read_text()) if args.warmup_workload else workload)
    torch.cuda.reset_peak_memory_stats()
    if args.profile_cuda:
        torch.cuda.profiler.start()
    results = []
    for i in range(args.repetitions):
        print(f"Python measured run {i + 1}/{args.repetitions}", flush=True)
        value = run()
        print(json.dumps(value["stats"]), flush=True)
        results.append(value)
    if args.profile_cuda:
        torch.cuda.profiler.stop()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(engine="nano-vllm-python", precision="bfloat16",
        workload=workload["name"], settings={key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()}, seed=workload.get("seed", 0),
        warmups=args.warmups, prefix_cache_between_runs="warm" if args.warm_prefix_cache else "cold",
        model_load_seconds=load_seconds, torch_version=torch.__version__,
        torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        kv_cache_bytes=llm.model_runner.kv_cache.numel() * llm.model_runner.kv_cache.element_size(),
        reference_allocation_override="fixed KV block count only; source files unchanged",
        runs=results), indent=2))


if __name__ == "__main__":
    main()

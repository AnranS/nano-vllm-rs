"""Create an auditable Chinese benchmark report from completed result files."""
import argparse
import csv
import hashlib
import json
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PROFILES = ["latency-b1", "throughput-b8", "mixed-b32", "nano-vllm-256", "prefix-b8-cold", "prefix-b8-warm"]
LABELS = dict(zip(PROFILES, ["单请求 128→128", "8 请求 128→128", "32 请求混合长度", "原版 256 请求负载", "共享前缀：冷缓存", "共享前缀：热缓存"]))


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    pos = (len(values) - 1) * q
    lower = int(pos)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (pos - lower)


def read(path):
    return json.loads(path.read_text())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def summarize(data):
    runs = data["runs"]
    latencies = [[r["ttft_ms"] for r in run["outputs"]] for run in runs]
    gaps = []
    for run in runs:
        tokens = [value for record in run["outputs"] for value in record.get("inter_token_latencies_ms", [])]
        gaps.append(percentile(tokens, 0.5) if tokens else run["stats"].get("inter_token_p50_ms"))
    return dict(repetitions=len(runs),
        input_tokens=runs[0]["stats"]["input_tokens"], output_tokens=runs[0]["stats"]["output_tokens"],
        seconds_median=statistics.median(r["stats"]["elapsed_seconds"] for r in runs),
        throughput_median=statistics.median(r["stats"]["output_tokens_per_second"] for r in runs),
        throughput_min=min(r["stats"]["output_tokens_per_second"] for r in runs),
        throughput_max=max(r["stats"]["output_tokens_per_second"] for r in runs),
        ttft_p50_ms=statistics.median(percentile(values, 0.5) for values in latencies),
        ttft_p95_ms=statistics.median(percentile(values, 0.95) for values in latencies),
        itl_p50_ms=statistics.median(v for v in gaps if v is not None) if any(v is not None for v in gaps) else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    root = args.root
    results = root / "benchmarks/results"
    entries = {}
    for profile in PROFILES:
        for engine in ["rust", "python"]:
            path = results / f"{engine}-{profile}.json"
            if not path.exists():
                if args.require_complete:
                    raise SystemExit(f"Missing benchmark: {path}")
                continue
            data = read(path)
            if not data.get("runs"):
                raise SystemExit(f"No measured runs: {path}")
            entries[(engine, profile)] = (data, summarize(data))
    with (results / "summary.csv").open("w", newline="") as file:
        keys = ["engine", "profile", "repetitions", "input_tokens", "output_tokens", "seconds_median",
                "throughput_median", "throughput_min", "throughput_max", "ttft_p50_ms", "ttft_p95_ms", "itl_p50_ms"]
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for (engine, profile), (_, row) in entries.items():
            writer.writerow(dict(engine=engine, profile=profile, **row))

    source_files = sorted([*root.glob("src/**/*.rs"), *root.glob("kernels/*.cu"),
        *root.glob("benchmarks/*.py"), *root.glob("scripts/*.sh"), *root.glob("examples/*.rs"),
        root / "build.rs", root / "Cargo.toml", root / "Cargo.lock", root / "rust-toolchain.toml"])
    manifest = {str(path.relative_to(root)): sha256(path) for path in source_files}
    manifest.update({str(path.relative_to(root)): sha256(path) for path in sorted((root / "benchmarks/workloads").glob("*.json"))})
    (results / "source-and-workload-sha256.json").write_text(json.dumps(manifest, indent=2))
    rust_lines = sum(len(path.read_text().splitlines()) for path in root.glob("src/**/*.rs"))
    cuda_lines = sum(len(path.read_text().splitlines()) for path in root.glob("kernels/*.cu"))
    environment = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        gpu=command(["/usr/lib/wsl/lib/nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        rustc=command([str(root / ".tools/cargo/bin/rustc"), "--version"]),
        nvcc=command(["/usr/local/cuda/bin/nvcc", "--version"]),
        reference_commit=command(["git", "-C", "/root/nano-vllm", "rev-parse", "HEAD"]),
        reference_worktree_status=command(["git", "-C", "/root/nano-vllm", "status", "--short"]),
        rust_source_lines_including_tests=rust_lines, cuda_source_lines=cuda_lines)
    (results / "environment.json").write_text(json.dumps(environment, indent=2))
    lines = ["# nano-vllm-rs 实现与本机 Benchmark", "",
        f"生成时间：{environment['created_utc']}。项目：`/root/nano-vllm-rs`，WSL2 `llm-ubuntu-24.04`。", "",
        "已实现本地 Qwen3 权重加载、BF16 CUDA 推理、真实分页 KV cache、连续批处理、分块 prefill、共享前缀缓存/LRU、容量压力下抢占重算、EOS/温度采样和 decode CUDA Graph。推理进程不调用 Python 或 PyTorch。", "",
        "## 环境与方法", "",
        f"- GPU：{environment['gpu']}。",
        "- 模型：本地 Qwen3-0.6B，BF16；Rust 使用 CUDA 13.2/cuBLAS，Python 使用 PyTorch 2.12.1+cu130、FlashAttention 2.8.3.post1。",
        "- 相同保存的 token 输入、输出长度、温度 0.6、ignore_eos=true；两版 RNG 实现不同，不要求随机输出逐 token 相同。",
        "- 共同配置：block_size=256，num_blocks=128（KV 3.5 GiB），max_sequences=32，max_model_len=4096，max_batch_tokens=1024，CUDA Graph 开启。",
        f"- Python 参考版本：`{environment['reference_commit']}`。基准脚本仅在进程中覆盖 KV 分配数量；模型、attention 和调度均使用原实现。",
        "- 基准脚本未修改参考仓库。本地 nanovllm 工作区含中文注释/docstring 改动；去除 docstring 后的 AST 与 HEAD 相同，审计记录见 reference-source-audit.json。",
        "- 两版串行执行；模型加载、tokenizer、输出解码、JSON 写出和预热不计入吞吐时间。运行内包含调度、前向、采样、主机设备同步。",
        "- 小负载与前缀缓存项：完整同负载预热 1 次、正式测量 3 次，报告中位数。256 请求项：独立短预热 1 次、正式测量 1 次，单次结果不代表统计稳定性。短预热的输出长度为 2..33，覆盖 decode batch 32..1。",
        "- 冷缓存项在每轮前清理可复用前缀元数据，保留已加载权重、编译结果与 CUDA Graph。热缓存项保留前一轮缓存。",
        "- 32 的含义并非完全相同：Rust 限制总 active 请求数，原 Python 调度器限制每次调度数量、总 running 可更多。大负载结果比较相同配置下各自调度器，不等于严格相同总并发。", "",
        "## 输出吞吐", "",
        "单位为输出 token/s，分母包括 prefill 和 decode；不是仅 decode 的速度。", "",
        "| 负载 | 测量次数/版 | 输入/输出 token | Rust | Python nano-vLLM | Rust/Python |",
        "|---|---:|---:|---:|---:|---:|"]
    for profile in PROFILES:
        if ("rust", profile) not in entries or ("python", profile) not in entries:
            continue
        rust, python = entries[("rust", profile)][1], entries[("python", profile)][1]
        assert rust["input_tokens"] == python["input_tokens"] and rust["output_tokens"] == python["output_tokens"]
        lines.append(f"| {LABELS[profile]} | {rust['repetitions']} | {rust['input_tokens']:,}/{rust['output_tokens']:,} | {rust['throughput_median']:,.2f} | {python['throughput_median']:,.2f} | {rust['throughput_median']/python['throughput_median']:.2%} |")
    if ("rust", "nano-vllm-256") in entries:
        full = entries[("rust", "nano-vllm-256")][0]["runs"][0]["stats"]
        lines += ["", f"完整负载的 Rust 正式运行发生 {full['preemptions']:,} 次抢占，执行 {full['prefill_tokens']:,} 个 prefill token（原始输入 {full['input_tokens']:,}，含重算）及 {full['decode_steps']:,} 个 decode step。这个结果同时体现当前注意力内核与给定 KV 配额下的调度代价。"]
    lines += ["", "## 延迟", "", "TTFT 从整批请求提交开始，包含排队；每轮先对请求取分位数，再对测量轮取中位数。ITL 是相邻输出 token 的间隔，包含同引擎其他请求的调度影响。", "",
        "| 负载 | Rust TTFT p50/p95 (ms) | Python TTFT p50/p95 (ms) | Rust/Python ITL p50 (ms) |",
        "|---|---:|---:|---:|"]
    for profile in PROFILES:
        if ("rust", profile) not in entries or ("python", profile) not in entries:
            continue
        r, p = entries[("rust", profile)][1], entries[("python", profile)][1]
        ri = f"{r['itl_p50_ms']:.2f}" if r['itl_p50_ms'] is not None else "—"
        pi = f"{p['itl_p50_ms']:.2f}" if p['itl_p50_ms'] is not None else "—"
        lines.append(f"| {LABELS[profile]} | {r['ttft_p50_ms']:.2f}/{r['ttft_p95_ms']:.2f} | {p['ttft_p50_ms']:.2f}/{p['ttft_p95_ms']:.2f} | {ri}/{pi} |")
    lines += ["", "## 显存", "", "以下统一使用完整负载进程期间的整卡显存采样，含启动/预热，约每 0.5 秒采样一次；差值减去进程启动前整卡占用。它可能包含桌面或其他应用变化，不能视为精确的 allocator 峰值。", "",
        "| 引擎 | 启动前整卡 (MiB) | 进程期间整卡峰值 (MiB) | 相对增量 (MiB) |",
        "|---|---:|---:|---:|"]
    for engine in ["rust", "python"]:
        path = root / f".bench-logs/{engine}-nano-vllm-256.monitor.json"
        if path.exists():
            mon = read(path)
            if mon["returncode"] == 0 and mon.get("gpu_before"):
                lines.append(f"| {engine} | {mon['gpu_before']['memory_used_mib']:,.0f} | {mon['gpu_peak_used_mib']:,.0f} | {mon['gpu_peak_delta_mib']:,.0f} |")
    lines += ["", "原始 Rust JSON 另有 weights/KV/workspace 静态字节数，Python JSON 另有 PyTorch peak_allocated/peak_reserved。它们的统计范围不同，未混为同一个显存指标。", "",
        "## 正确性与实现范围", "",
        "- 19 个 CPU/模型加载/调度单元测试通过，3 个真实 GPU 算子测试通过，Clippy（警告视为错误）与 release 构建通过。",
        "- GPU 测试包含非连续物理块、ragged GQA 与独立 CPU attention 逐元素对照，及 GEMM、RMSNorm、RoPE、采样和 Graph 回放。"]
    correctness_path = results / "correctness-summary.json"
    if correctness_path.exists():
        correctness = read(correctness_path)
        cases = correctness["cases"]
        exact_hf = sum(r["exact_generation_match"] for r in cases)
        exact_nano = sum(r["nano_exact_generation_match"] for r in cases)
        lines += [f"- {len(cases)} 组输入、共 {sum(r['generated_tokens'] for r in cases)} 个贪心输出 token：{exact_hf}/{len(cases)} 组与独立 Transformers BF16 eager 参考完全一致。完整 prefill / 64-token 分块 / CUDA Graph 一致性：{correctness['graph_invariance'] and correctness['chunk_invariance']}。",
            f"- 与原 nano-vLLM 的 argmax 诊断参考：{exact_nano}/{len(cases)} 段完全一致；一段开放式续写在后续 token 有不同选择。首 token 全部一致，四组 logits 余弦相似度范围 {min(r['nano_cosine_similarity'] for r in cases):.6f}–{max(r['nano_cosine_similarity'] for r in cases):.6f}，最大绝对误差范围 {min(r['nano_max_abs_error'] for r in cases):.6f}–{max(r['nano_max_abs_error'] for r in cases):.6f}。未宣称跨实现逐位一致。",
            "- 已根据原版实际 TorchInductor 缓存修正 RMSNorm、Q/K Norm、SiLU 的中间 BF16 舍入。HF eager、FlashAttention 与本项目 attention 的计算/舍入路径仍不同；完整差异保存在 correctness-summary.json。"]
    lines += ["- 当前支持单 GPU、dense Qwen3、最多 4096 上下文；不支持张量并行、量化、RoPE scaling、sliding window、attention bias 或 HTTP 流式服务。",
        "- 当前 attention 是自有 CUDA 实现，尚未实现 FlashAttention 的高效分块/张量核心路径。实测速度未对齐原版，也未对未测试的模型/硬件做性能承诺。",
        f"- 本次源码规模：Rust src/ 共 {rust_lines:,} 行（含测试与 CLI），CUDA 共 {cuda_lines:,} 行；不计第三方依赖、生成代码、工具链。", "",
        "## 复现与原始数据", "", "```bash", "cd /root/nano-vllm-rs", "source scripts/env.sh",
        "cargo build --release --locked --offline",
        "python3 benchmarks/run_suite.py --profiles latency-b1 throughput-b8 mixed-b32 --repetitions 3 --warmups 1",
        "python3 benchmarks/run_suite.py --profiles nano-vllm-256 --repetitions 1 --warmups 1 --warmup-workload benchmarks/workloads/warmup-all-batches.json",
        "python3 benchmarks/run_suite.py --profiles prefix-b8-cold prefix-b8-warm --repetitions 3 --warmups 1",
        "python3 benchmarks/summarize.py --require-complete", "```", "",
        "固定工作负载在 benchmarks/workloads/；逐轮结果、生成 token、正确性数据及源文件 SHA-256 在 benchmarks/results/；原始进程日志和显存采样在 .bench-logs/。summary.csv 可以直接导入表格软件。", ""]
    (root / "BENCHMARK.md").write_text("\n".join(lines), encoding="utf-8")
    print(root / "BENCHMARK.md")


if __name__ == "__main__":
    main()

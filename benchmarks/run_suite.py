"""Run engines sequentially with identical saved workloads and GPU monitoring."""
import argparse
import json
import os
import statistics
import subprocess
import shutil
import threading
import time
from pathlib import Path


def gpu_snapshot():
    try:
        tool = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
        out = subprocess.check_output([tool, "--query-gpu=name,memory.used,temperature.gpu,power.draw,clocks.sm,utilization.gpu",
                                       "--format=csv,noheader,nounits"], text=True, timeout=5)
        cells = [s.strip() for s in out.strip().splitlines()[0].split(",")]
        return dict(timestamp=time.time(), name=cells[0], memory_used_mib=float(cells[1]),
                    temperature_c=float(cells[2]), power_w=float(cells[3]),
                    sm_clock_mhz=float(cells[4]), utilization_percent=float(cells[5]))
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def run_monitored(command, log_path, env):
    baseline = gpu_snapshot()
    samples = [baseline] if baseline else []
    stopped = threading.Event()

    def monitor():
        while not stopped.wait(0.5):
            value = gpu_snapshot()
            if value:
                samples.append(value)

    worker = threading.Thread(target=monitor, daemon=True)
    worker.start()
    started = time.time()
    try:
        with log_path.open("w") as log:
            process = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    finally:
        stopped.set()
        worker.join(timeout=6)
    peak = max((s["memory_used_mib"] for s in samples), default=None)
    record = dict(command=command, returncode=process.returncode, process_wall_seconds=time.time() - started,
                  device_memory_note="device-wide polling includes other applications and startup; not a CUDA allocator high-water mark",
                  gpu_before=baseline, gpu_peak_used_mib=peak,
                  gpu_peak_delta_mib=peak - baseline["memory_used_mib"] if baseline and peak else None,
                  samples=samples)
    log_path.with_suffix(".monitor.json").write_text(json.dumps(record, indent=2))
    if process.returncode:
        print(log_path.read_text()[-6000:], flush=True)
        raise SystemExit(f"Command failed; see {log_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/root/huggingface/Qwen3-0.6B"))
    parser.add_argument("--python", type=Path, default=Path("/root/nano-vllm/.venv/bin/python"))
    parser.add_argument("--engines", nargs="+", default=["rust", "python"], choices=["rust", "python"])
    parser.add_argument("--profiles", nargs="+", default=["latency-b1", "throughput-b8", "mixed-b32", "nano-vllm-256", "prefix-b8-cold", "prefix-b8-warm"])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--warmup-workload", type=Path)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    results = root / "benchmarks/results"
    results.mkdir(parents=True, exist_ok=True)
    logs = root / ".bench-logs"
    logs.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false",
               TORCHINDUCTOR_CACHE_DIR=str(root / ".cache/torchinductor"),
               TRITON_CACHE_DIR=str(root / ".cache/triton"))
    for profile in args.profiles:
        workload = "prefix-b8" if profile.startswith("prefix-b8-") else profile
        for engine in args.engines:
            name = f"{engine}-{profile}{args.suffix}"
            command = ([str(root / "target/release/nano-vllm-rs"), "bench"] if engine == "rust" else
                       [str(args.python), "benchmarks/python_reference.py"])
            command += ["--model", str(args.model), "--workload", f"benchmarks/workloads/{workload}.json",
                        "--output", str(results / f"{name}.json"),
                        "--repetitions", str(args.repetitions), "--warmups", str(args.warmups),
                        "--block-size", "256", "--num-blocks", "128", "--max-sequences", "32",
                        "--max-model-len", "4096", "--max-batch-tokens", "1024"]
            if profile.endswith("-warm"):
                command.append("--warm-prefix-cache")
            if args.eager:
                command.append("--eager")
            if args.warmup_workload:
                command += ["--warmup-workload", str(args.warmup_workload)]
            print(f"START {name}", flush=True)
            run_monitored(command, logs / f"{name}.log", env)
            data = json.loads((results / f"{name}.json").read_text())
            throughputs = [r["stats"]["output_tokens_per_second"] for r in data["runs"]]
            print(f"DONE {name}: median {statistics.median(throughputs):.2f} output tok/s; runs {throughputs}", flush=True)


if __name__ == "__main__":
    main()

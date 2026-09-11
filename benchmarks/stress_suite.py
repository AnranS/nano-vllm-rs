"""Repeat a fixed real workload within one model process per backend.

Default: native, four nano-vllm-256 repetitions, approximately five minutes of
measured generation on the recorded host. This is an estimate, not a timeout.
Set --repetitions explicitly for another backend or machine. Model loading and
one short warm-up are additional wall time. Engines always run serially.

Examples:
  python benchmarks/stress_suite.py --backends flash --repetitions 10 --output-dir benchmarks/stress-flash
  python benchmarks/stress_suite.py --backends native --include-python --repetitions 3 --output-dir benchmarks/stress-reference
  python benchmarks/stress_suite.py --self-test

No environment dump, remote URLs, credentials, or git configuration are recorded.
Current source identity is not represented as proof of the binary's build commit:
the binary SHA-256 is authoritative for identifying the executable actually used.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import shutil
import subprocess
import sys
import time


def read_json(path):
    def invalid(value):
        raise ValueError(f"Nonfinite JSON number: {value}")
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream, parse_constant=invalid)
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=invalid)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def archive_new_result(path, keep_raw=False):
    """Losslessly archive only a result in this invocation's newly created directory."""
    path = Path(path)
    destination = path.with_name(path.name + ".gz")
    identity = dict(uncompressed_sha256=sha256(path), uncompressed_bytes=path.stat().st_size,
                    uncompressed_path=str(path))
    with path.open("rb") as source, destination.open("xb") as target:
        with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as archive:
            shutil.copyfileobj(source, archive, 1024 * 1024)
    restored = hashlib.sha256()
    with gzip.open(destination, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            restored.update(chunk)
    if restored.hexdigest() != identity["uncompressed_sha256"]:
        raise ValueError("Compressed result failed lossless SHA-256 verification; original retained")
    identity.update(compressed_path=str(destination), compressed_sha256=sha256(destination),
                    compressed_bytes=destination.stat().st_size,
                    decompressed_sha256_verified=True, uncompressed_retained=keep_raw)
    if not keep_raw:
        path.unlink()
    return identity


def nonnegative(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def percentile(values, quantile):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def distribution(values):
    if not values:
        return {"count": 0, "median": None, "p95": None, "min": None, "max": None,
                "mean": None, "population_stddev": None, "coefficient_of_variation_percent": None}
    mean = statistics.mean(values)
    deviation = statistics.pstdev(values)
    return dict(count=len(values), median=statistics.median(values), p95=percentile(values, .95),
                min=min(values), max=max(values), mean=mean, population_stddev=deviation,
                coefficient_of_variation_percent=100 * deviation / mean if mean else None)


def validate_workload(workload, max_model_len):
    rows = workload.get("requests")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Workload must contain a nonempty requests list")
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Every request must be an object")
        request_id, tokens, count = row.get("id"), row.get("prompt_token_ids"), row.get("max_tokens")
        if not integer(request_id) or request_id < 0 or request_id in seen:
            raise ValueError("Request IDs must be distinct nonnegative integers")
        seen.add(request_id)
        if not isinstance(tokens, list) or not tokens or any(not integer(t) or t < 0 or t > 0xffffffff for t in tokens):
            raise ValueError(f"Invalid prompt token list for request {request_id}")
        if not integer(count) or count <= 0 or len(tokens) + count > max_model_len:
            raise ValueError(f"Invalid output length/context limit for request {request_id}")
        if row.get("ignore_eos") is not True:
            raise ValueError("Fixed-length stress requires explicit ignore_eos=true for every request")
        if not nonnegative(row.get("temperature", 0)):
            raise ValueError(f"Invalid temperature for request {request_id}")
    if not integer(workload.get("seed", 0)) or workload.get("seed", 0) < 0:
        raise ValueError("Seed must be a nonnegative integer")
    return {row["id"]: row for row in rows}


def analyze_run(run, expected, repetition, python=False):
    errors, checksum_rows, ttfts, gaps = [], [], [], []
    outputs, stats = run.get("outputs"), run.get("stats")
    if not isinstance(outputs, list) or not isinstance(stats, dict):
        return dict(repetition=repetition, valid=False, errors=["Missing outputs list or stats object"])
    by_id = {}
    total = 0
    for row in outputs:
        if not isinstance(row, dict):
            errors.append("Non-object output")
            continue
        request_id = row.get("request_id") if python else row.get("id")
        if not integer(request_id) or request_id in by_id or request_id not in expected:
            errors.append(f"Invalid, duplicate, or unexpected request ID: {request_id!r}")
            continue
        by_id[request_id] = row
        tokens = row.get("token_ids")
        if not isinstance(tokens, list) or any(not integer(t) or t < 0 or t > 0xffffffff for t in tokens):
            errors.append(f"Invalid output tokens for request {request_id}")
            continue
        total += len(tokens)
        if len(tokens) != expected[request_id]["max_tokens"]:
            errors.append(f"Request {request_id}: expected {expected[request_id]['max_tokens']} tokens, got {len(tokens)}")
        checksum_rows.append(dict(id=request_id, token_ids=tokens))
        if row.get("finish_reason", "length") != "length":
            errors.append(f"Unexpected finish reason for request {request_id}")
        if "prompt_tokens" in row and row["prompt_tokens"] != len(expected[request_id]["prompt_token_ids"]):
            errors.append(f"Wrong prompt token count for request {request_id}")
        ttft, latency = row.get("ttft_ms"), row.get("latency_ms")
        if not nonnegative(ttft) or not nonnegative(latency) or latency < ttft:
            errors.append(f"Invalid request latency/TTFT for request {request_id}")
        else:
            ttfts.append(ttft)
        if not python:
            intervals = row.get("inter_token_latencies_ms")
            if (not isinstance(intervals, list) or len(intervals) != max(0, len(tokens) - 1)
                    or any(not nonnegative(x) for x in intervals)):
                errors.append(f"Invalid or missing raw ITL values for request {request_id}")
            else:
                gaps.extend(intervals)
    missing = sorted(set(expected) - set(by_id))
    if missing:
        errors.append(f"Missing request IDs: {missing[:20]} (total {len(missing)})")
    if len(outputs) != len(expected):
        errors.append(f"Expected {len(expected)} outputs, got {len(outputs)}")
    requested_total = sum(row["max_tokens"] for row in expected.values())
    input_total = sum(len(row["prompt_token_ids"]) for row in expected.values())
    for key, expected_count in [("output_tokens", requested_total), ("input_tokens", input_total)]:
        if not integer(stats.get(key)) or stats[key] != expected_count:
            errors.append(f"stats.{key}: expected {expected_count}, got {stats.get(key)!r}")
    if total != requested_total:
        errors.append(f"Actual output token total {total} differs from requested {requested_total}")
    if "total_requests" in stats and stats["total_requests"] != len(expected):
        errors.append("Incorrect stats.total_requests")
    elapsed, recorded_rate = stats.get("elapsed_seconds"), stats.get("output_tokens_per_second")
    if not nonnegative(elapsed) or elapsed <= 0:
        errors.append("elapsed_seconds must be finite and positive")
        calculated_rate = None
    else:
        calculated_rate = total / elapsed
        if not nonnegative(recorded_rate) or not math.isclose(recorded_rate, calculated_rate, rel_tol=1e-9, abs_tol=1e-6):
            errors.append("Recorded throughput does not equal actual output token count / elapsed_seconds")
    if python:
        # The Python reference does NOT emit token timestamps or raw intervals.
        # Its stats compute these quantiles from all observed token gaps.
        itl50, itl95 = stats.get("inter_token_p50_ms"), stats.get("inter_token_p95_ms")
        has_gaps = any(row["max_tokens"] > 1 for row in expected.values())
        if has_gaps and (not nonnegative(itl50) or not nonnegative(itl95) or itl95 < itl50):
            errors.append("Missing/invalid Python-reported ITL quantiles")
        itl_source = "Python stats.inter_token_p50_ms/p95_ms; no raw intervals exported"
    else:
        itl50, itl95 = percentile(gaps, .5), percentile(gaps, .95)
        itl_source = "Recomputed from all RequestOutput.inter_token_latencies_ms in this repetition"
    checksum = canonical_sha(sorted(checksum_rows, key=lambda row: row["id"])) if len(checksum_rows) == len(expected) else None
    return dict(repetition=repetition, valid=not errors, errors=errors,
                output_tokens=total, elapsed_seconds=elapsed,
                output_tokens_per_second=calculated_rate,
                generated_tokens_sha256=checksum, ttft_p50_ms=percentile(ttfts, .5),
                ttft_p95_ms=percentile(ttfts, .95), inter_token_p50_ms=itl50,
                inter_token_p95_ms=itl95, inter_token_quantile_source=itl_source,
                raw_itl_count=len(gaps) if not python else None,
                prefill_tokens=stats.get("prefill_tokens"), decode_steps=stats.get("decode_steps"),
                preemptions=stats.get("preemptions"), cached_tokens=stats.get("cached_tokens"))


def analyze_document(document, expected, repetitions, python=False):
    errors = []
    runs = document.get("runs")
    if not isinstance(runs, list):
        runs = []
        errors.append("Missing runs list")
    if len(runs) != repetitions:
        errors.append(f"Expected {repetitions} measured repetitions, got {len(runs)}")
    rows = [analyze_run(run, expected, i + 1, python) if isinstance(run, dict)
            else dict(repetition=i + 1, valid=False, errors=["Non-object run"])
            for i, run in enumerate(runs)]
    checksums = [row.get("generated_tokens_sha256") for row in rows]
    stable = (len(rows) >= 2 and all(checksums) and len(set(checksums)) == 1)
    changed = [i + 1 for i, checksum in enumerate(checksums) if checksum != checksums[0]] if checksums else []
    if len(rows) >= 2 and not stable:
        errors.append("Repeated output checksums are different or unavailable")
    valid_rows = [row for row in rows if row["valid"]]
    metrics = {key: distribution([row[key] for row in valid_rows if nonnegative(row.get(key))])
               for key in ["output_tokens_per_second", "ttft_p50_ms", "ttft_p95_ms",
                           "inter_token_p50_ms", "inter_token_p95_ms"]}
    return dict(passed=not errors and len(valid_rows) == repetitions, errors=errors,
                completed_repetitions=len(rows), valid_repetitions=len(valid_rows),
                repeated_output_checksum_consistent=stable if len(rows) >= 2 else None,
                checksum_comparison_note="Within this backend with the same saved seed and cold reusable prefix metadata; never across backends",
                checksum_differs_from_first_repetitions=changed,
                measured_seconds=sum(row["elapsed_seconds"] for row in valid_rows),
                runs=rows, distributions_across_valid_repetitions=metrics,
                quantile_aggregation="TTFT/ITL are per-repetition quantiles, then summarized across repetitions; not pooled multi-run quantiles")


def memory_trend(monitor):
    samples = [s for s in monitor.get("samples", []) if isinstance(s, dict)
               and nonnegative(s.get("timestamp")) and nonnegative(s.get("memory_used_mib"))]
    samples.sort(key=lambda s: s["timestamp"])
    result = dict(available=bool(samples), sample_count=len(samples),
                  scope="Device-wide approximately 0.5-second polling across process startup, warm-up and measurement",
                  interpretation="Descriptive trend, not a process allocator high-water mark or a memory-leak verdict; other applications may contribute",
                  per_repetition_memory_available=False)
    if not samples:
        return result
    values = [s["memory_used_mib"] for s in samples]
    window = max(1, len(values) // 4)
    tail = samples[len(samples) // 2:]
    x = [(s["timestamp"] - samples[0]["timestamp"]) / 60 for s in tail]
    y = [s["memory_used_mib"] for s in tail]
    mean_x, mean_y = statistics.mean(x), statistics.mean(y)
    denominator = sum((v - mean_x) ** 2 for v in x)
    slope = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y)) / denominator if denominator else None
    result.update(first_mib=values[0], last_mib=values[-1], min_mib=min(values), peak_mib=max(values),
                  observed_seconds=samples[-1]["timestamp"] - samples[0]["timestamp"],
                  last_minus_first_mib=values[-1] - values[0],
                  first_quarter_median_mib=statistics.median(values[:window]),
                  last_quarter_median_mib=statistics.median(values[-window:]),
                  last_quarter_range_mib=max(values[-window:]) - min(values[-window:]),
                  latter_half_linear_slope_mib_per_minute=slope)
    return result


def git_identity(root):
    def command(args):
        try:
            result = subprocess.run(["git", "-C", str(root), *args], check=False,
                                    capture_output=True, text=True, timeout=10)
            return result.stdout.strip() if result.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None
    status = command(["status", "--porcelain", "--untracked-files=no"])
    return dict(commit=command(["rev-parse", "HEAD"]),
                tracked_worktree_dirty=bool(status) if status is not None else None)


def source_identity(root):
    files = [root / name for name in ["Cargo.toml", "Cargo.lock", "build.rs", "rust-toolchain.toml",
             "kernels/kernels.cu", "kernels/flash_native/flash_native.cu",
             "kernels/flash_native/flash_native.h", "kernels/flash_native/vendor-sha256.json",
             "benchmarks/python_reference.py", "benchmarks/run_suite.py", "benchmarks/stress_suite.py"]]
    files.extend(sorted((root / "src").rglob("*.rs")))
    manifest = {str(path.relative_to(root)): sha256(path) for path in files if path.is_file()}
    return dict(**git_identity(root), selected_source_files_sha256=manifest,
                selected_source_manifest_sha256=canonical_sha(manifest),
                note="Checkout at invocation; not an attestation of executable build commit. Binary SHA-256 separately identifies the executable.")


def resolve(root, path):
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def summarize_engine(engine, raw_path, monitor_path, expected, args):
    result = dict(backend=engine, result_path=str(raw_path), monitor_path=str(monitor_path))
    errors = []
    try:
        document = read_json(raw_path)
        result.update(analyze_document(document, expected, args.repetitions, engine == "python"))
        if document.get("precision") != "bfloat16":
            errors.append("Expected recorded BF16 precision")
        if document.get("seed") != args.workload_data.get("seed", 0):
            errors.append("Recorded seed differs from workload")
        if document.get("warmups") != args.warmups:
            errors.append("Recorded warm-up count differs from requested count")
        if document.get("prefix_cache_between_runs") != "cold":
            errors.append("Stress comparison requires cold reusable prefix metadata between repetitions")
        settings = document.get("settings", {})
        for name in ["block_size", "num_blocks", "max_sequences", "max_model_len", "max_batch_tokens"]:
            if settings.get(name) != getattr(args, name):
                errors.append(f"Recorded setting {name} differs from command")
        if engine != "python" and settings.get("attention_backend") != engine:
            errors.append("Recorded attention backend differs from command")
        result["reported_model_load_seconds"] = document.get("model_load_seconds")
        result["reported_version"] = document.get("torch_version") if engine == "python" else document.get("version")
        result["model_memory_static_bytes"] = document.get("model_memory")
        result["torch_peak_allocated_bytes"] = document.get("torch_peak_allocated_bytes")
        result["torch_peak_reserved_bytes"] = document.get("torch_peak_reserved_bytes")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result.update(passed=False, errors=[f"Cannot validate saved result: {type(exc).__name__}: {exc}"])
    try:
        monitor = read_json(monitor_path)
        result["gpu_memory_trend"] = memory_trend(monitor)
        result["process_returncode"] = monitor.get("returncode")
        result["process_wall_seconds"] = monitor.get("process_wall_seconds")
        if monitor.get("returncode") != 0:
            errors.append(f"Backend process return code: {monitor.get('returncode')!r}")
    except (OSError, ValueError, TypeError) as exc:
        result["gpu_memory_trend"] = dict(available=False, reason=f"{type(exc).__name__}: {exc}")
        errors.append("Missing or invalid process monitor record")
    result.setdefault("errors", []).extend(errors)
    result["passed"] = result.get("passed", False) and not errors
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--backends", nargs="+", choices=["native", "flash"], default=["native"])
    parser.add_argument("--include-python", action="store_true")
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/stress-results"))
    parser.add_argument("--workload", type=Path, default=Path("benchmarks/workloads/nano-vllm-256.json"))
    parser.add_argument("--warmup-workload", type=Path, default=Path("benchmarks/workloads/warmup-all-batches.json"))
    parser.add_argument("--model", type=Path, default=Path("/root/huggingface/Qwen3-0.6B"))
    parser.add_argument("--binary", type=Path, default=Path("target/release/nano-vllm-rs"))
    parser.add_argument("--python", type=Path, default=Path("/root/nano-vllm/.venv/bin/python"))
    parser.add_argument("--reference", type=Path, default=Path("/root/nano-vllm"))
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=128)
    parser.add_argument("--max-sequences", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-batch-tokens", type=int, default=1024)
    parser.add_argument("--keep-raw", action="store_true", help="Keep this invocation's uncompressed JSON as well as verified lossless .json.gz")
    parser.add_argument("--self-test", action="store_true", help="Run CPU-only validator tests; never import CUDA/Torch or launch an engine")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.repetitions < 2:
        parser.error("Stress stability requires at least two repetitions in the same model process")
    if args.warmups < 0 or any(getattr(args, key) <= 0 for key in ["block_size", "num_blocks", "max_sequences", "max_model_len", "max_batch_tokens"]):
        parser.error("Warmups must be nonnegative and resource limits must be positive")
    if len(args.backends) != len(set(args.backends)):
        parser.error("Do not list a backend more than once")
    root = args.root.resolve()
    for key in ["output_dir", "workload", "warmup_workload", "model", "binary", "reference"]:
        setattr(args, key, resolve(root, getattr(args, key)))
    # Preserve venv/bin/python's symlink spelling: resolving it can bypass venv discovery.
    args.python = (args.python if args.python.is_absolute() else root / args.python).absolute()
    args.workload_data = read_json(args.workload)
    expected = validate_workload(args.workload_data, args.max_model_len)
    validate_workload(read_json(args.warmup_workload), args.max_model_len)
    if not args.binary.is_file():
        parser.error(f"Build the Rust executable first: {args.binary}")
    if args.include_python and not args.python.is_file():
        parser.error(f"Python reference interpreter not found: {args.python}")
    if not (args.model / "config.json").is_file():
        parser.error("Model config.json is missing")
    if args.output_dir.exists():
        parser.error(f"Refusing to overwrite {args.output_dir}; use a new --output-dir")
    os.chdir(root)
    # Import only at execution time; --self-test has no run_suite/GPU dependency.
    sys.path.insert(0, str(root / "benchmarks"))
    from run_suite import run_monitored
    args.output_dir.mkdir(parents=True)
    logs = args.output_dir / "logs"
    logs.mkdir()
    engines = args.backends + (["python"] if args.include_python else [])
    report = dict(format_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
                  status="running", requested_backends=engines, requested_repetitions=args.repetitions,
                  seed=args.workload_data.get("seed", 0), workload_path=str(args.workload),
                  workload_sha256=sha256(args.workload), workload_name=args.workload_data.get("name"),
                  requests_per_repetition=len(expected),
                  expected_input_tokens_per_repetition=sum(len(r["prompt_token_ids"]) for r in expected.values()),
                  expected_output_tokens_per_repetition=sum(r["max_tokens"] for r in expected.values()),
                  warmup_workload_path=str(args.warmup_workload), warmup_workload_sha256=sha256(args.warmup_workload),
                  model_path=str(args.model), model_config_sha256=sha256(args.model / "config.json"),
                  build_identity=source_identity(root),
                  rust_binary=dict(path=str(args.binary), sha256=sha256(args.binary)),
                  settings={key: getattr(args, key) for key in ["warmups", "block_size", "num_blocks", "max_sequences", "max_model_len", "max_batch_tokens"]},
                  methodology=dict(one_model_process_per_backend=True, engines_serial=True,
                      precision="bfloat16", cuda_graphs=True, profiler_enabled=False,
                      reusable_prefix_cache_between_repetitions="cold",
                      throughput="actual output tokens / measured elapsed seconds including prefill, decode, scheduling, transfers and synchronization; excludes loading and separate warm-up",
                      ttft="From whole-batch submission, including queueing",
                      checksum="SHA-256 of canonical ID-sorted {id, token_ids} outputs, excluding timings",
                      rng_comparison="Same-backend repeated seed only; Python and Rust RNGs differ",
                      default_duration_note="Four native repetitions approximate five measured minutes on the recorded host; no duration guarantee or timeout"),
                  backends=[])
    if args.include_python:
        report["python_reference"] = dict(path=str(args.reference), **git_identity(args.reference),
                                          interpreter_path=str(args.python), interpreter_sha256=sha256(args.python))
    summary_path = args.output_dir / "summary.json"
    write_json(summary_path, report)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false",
                       TORCHINDUCTOR_CACHE_DIR=str(root / ".cache/torchinductor"),
                       TRITON_CACHE_DIR=str(root / ".cache/triton"))
    started = time.monotonic()
    for engine in engines:
        raw = args.output_dir / f"{engine}.json"
        log = logs / f"{engine}.log"
        monitor = log.with_suffix(".monitor.json")
        command = ([str(args.python), "benchmarks/python_reference.py", "--reference", str(args.reference)]
                   if engine == "python" else [str(args.binary), "bench", "--attention-backend", engine])
        command += ["--model", str(args.model), "--workload", str(args.workload),
                    "--output", str(raw), "--warmups", str(args.warmups),
                    "--warmup-workload", str(args.warmup_workload), "--repetitions", str(args.repetitions)]
        for key in ["block_size", "num_blocks", "max_sequences", "max_model_len", "max_batch_tokens"]:
            command += ["--" + key.replace("_", "-"), str(getattr(args, key))]
        print(f"START {engine}: {args.repetitions} repetitions in one model process", flush=True)
        process_error, interrupted = None, False
        try:
            run_monitored(command, log, environment)
        except KeyboardInterrupt:
            process_error, interrupted = "Interrupted by user", True
        except (Exception, SystemExit) as exc:
            process_error = f"{type(exc).__name__}: {exc}"
        result = summarize_engine(engine, raw, monitor, expected, args)
        result["command"] = command
        if process_error:
            result["errors"].append(process_error)
            result["passed"] = False
        if raw.is_file():
            try:
                result["raw_result_archive"] = archive_new_result(raw, args.keep_raw)
                result["result_path"] = result["raw_result_archive"]["compressed_path"]
            except (OSError, ValueError) as exc:
                result["errors"].append(f"Cannot archive new raw result: {type(exc).__name__}: {exc}")
                result["passed"] = False
        report["backends"].append(result)
        report["suite_process_wall_seconds"] = time.monotonic() - started
        report["status"] = "interrupted" if interrupted else "running"
        write_json(summary_path, report)
        median = result.get("distributions_across_valid_repetitions", {}).get("output_tokens_per_second", {}).get("median")
        print(f"DONE {engine}: passed={result['passed']}, median output tok/s={median}", flush=True)
        if interrupted:
            return 130
    report["status"] = "passed" if all(row["passed"] for row in report["backends"]) else "failed"
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(summary_path, report)
    print(f"Stress {report['status']}: {summary_path}", flush=True)
    return 0 if report["status"] == "passed" else 1


def self_test():
    """Exercise failures as well as native and Python schemas, entirely on CPU."""
    import copy
    import unittest

    class ValidationTests(unittest.TestCase):
        def setUp(self):
            self.workload = dict(seed=0, requests=[dict(id=3, prompt_token_ids=[10, 11],
                                                       max_tokens=3, ignore_eos=True, temperature=.6)])
            self.expected = validate_workload(self.workload, 4096)
            self.run = dict(outputs=[dict(id=3, token_ids=[20, 21, 22], ttft_ms=1., latency_ms=6.,
                                         inter_token_latencies_ms=[2., 3.], finish_reason="length")],
                            stats=dict(output_tokens=3, input_tokens=2, elapsed_seconds=.01,
                                       output_tokens_per_second=300., total_requests=1))

        def test_native_stable_and_actual_quantiles(self):
            value = analyze_document(dict(runs=[self.run, copy.deepcopy(self.run)]), self.expected, 2)
            self.assertTrue(value["passed"])
            self.assertTrue(value["repeated_output_checksum_consistent"])
            self.assertEqual(value["runs"][0]["inter_token_p50_ms"], 2.5)
            self.assertAlmostEqual(value["runs"][0]["inter_token_p95_ms"], 2.95)

        def test_python_reported_itl(self):
            row = self.run["outputs"][0]
            row["request_id"] = row.pop("id")
            row.pop("inter_token_latencies_ms")
            row["mean_tpot_ms"] = 999  # Never used to invent quantiles.
            self.run["stats"].update(inter_token_p50_ms=2.5, inter_token_p95_ms=2.95)
            value = analyze_run(self.run, self.expected, 1, python=True)
            self.assertTrue(value["valid"])
            self.assertEqual(value["inter_token_p50_ms"], 2.5)
            self.assertIsNone(value["raw_itl_count"])

        def test_malformed_counts_and_ids_fail(self):
            self.run["outputs"].append(copy.deepcopy(self.run["outputs"][0]))
            value = analyze_run(self.run, self.expected, 1)
            self.assertFalse(value["valid"])
            self.run["outputs"].pop()
            self.run["outputs"][0]["token_ids"].pop()
            self.assertFalse(analyze_run(self.run, self.expected, 1)["valid"])

        def test_bad_throughput_and_missing_repetition_fail(self):
            self.run["stats"]["output_tokens_per_second"] = 123.
            self.assertFalse(analyze_run(self.run, self.expected, 1)["valid"])
            self.assertFalse(analyze_document(dict(runs=[]), self.expected, 2)["passed"])

        def test_checksum_instability_fails_without_cross_backend_check(self):
            second = copy.deepcopy(self.run)
            second["outputs"][0]["token_ids"][1] += 1
            value = analyze_document(dict(runs=[self.run, second]), self.expected, 2)
            self.assertFalse(value["passed"])
            self.assertEqual(value["checksum_differs_from_first_repetitions"], [2])

        def test_nonfinite_and_missing_itl_fail(self):
            self.run["outputs"][0]["ttft_ms"] = math.nan
            self.assertFalse(analyze_run(self.run, self.expected, 1)["valid"])
            self.run["outputs"][0]["ttft_ms"] = 1.
            self.run["outputs"][0].pop("inter_token_latencies_ms")
            self.assertFalse(analyze_run(self.run, self.expected, 1)["valid"])

        def test_memory_trend_is_descriptive(self):
            value = memory_trend(dict(samples=[dict(timestamp=60*i, memory_used_mib=100+i) for i in range(6)]))
            self.assertEqual(value["latter_half_linear_slope_mib_per_minute"], 1.)
            self.assertFalse(value["per_repetition_memory_available"])
            self.assertFalse(memory_trend({})["available"])

        def test_workload_requires_fixed_length(self):
            self.workload["requests"][0]["ignore_eos"] = False
            with self.assertRaises(ValueError):
                validate_workload(self.workload, 4096)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ValidationTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    raise SystemExit(main())

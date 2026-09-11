"""Additional numerical diagnostics for an attention backend change.

This does not replace compare_correctness.py or relax its original 64-token
acceptance. Run that script on this build's outputs as well. This suite compares
identical teacher-forced contexts, including the original sixth-token tie,
instead of treating diverged free-running generations as equal contexts.

Example (run GPU stages serially, never during a timing benchmark):
  python benchmarks/validate_attention.py prepare --suite benchmarks/attention-validation
  python benchmarks/validate_attention.py hf --suite benchmarks/attention-validation
  python benchmarks/validate_attention.py rust --suite benchmarks/attention-validation --label before --binary target/release/nano-vllm-rs-before
  python benchmarks/validate_attention.py rust --suite benchmarks/attention-validation --label after
  python benchmarks/validate_attention.py report --suite benchmarks/attention-validation --candidate after --baseline before --legacy-summary benchmarks/attention-results/correctness-summary.json

"prepare" and "report" never initialize a GPU. Per-case logits are compressed
float32 (~12 MiB per backend for 20 x 151936 values), not every sequence position.
The HF stage explicitly requests only the last position to bound memory usage.
"rust" runs native processes sequentially. Its logits are full-prefill eager;
graph/chunk/prefix checks compare generated tokens separately, without claiming
that the logits CLI exercises decode graphs. Diagnostic metrics intentionally
have no invented universal BF16 tolerance: reference disagreement is visible
and requires review, even when the reference top two logits are tied.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def new_directory(path):
    path = Path(path)
    if path.exists():
        raise ValueError(f"Refusing to overwrite {path}; use a new suite or label")
    path.mkdir(parents=True)
    return path


def prepare(args):
    original = read(args.fixtures)
    fixtures = original["results"]
    if len(fixtures) != 4 or any(len(row["token_ids"]) != 16 for row in fixtures):
        raise ValueError("Expected the original four 16-token Transformers fixtures")
    suite = new_directory(args.suite)
    cases = []

    def add(name, tokens, origin):
        cases.append(dict(id=len(cases), name=name, prompt_token_ids=tokens, origin=origin))

    for item in fixtures:
        add(f"legacy-{item['request_id']}-first", item["prompt_token_ids"],
            "Original fixed prompt; no new tokenization")
    for item in fixtures:
        add(f"legacy-{item['request_id']}-sixth",
            item["prompt_token_ids"] + item["token_ids"][:5],
            "Original reference's first five generated tokens are teacher-forced")
    # Valid natural-text IDs from the fixed long fixture: no tokenizer/model
    # changes can silently alter boundary lengths or shared prefix content.
    text = fixtures[3]["prompt_token_ids"]
    if not text:
        raise ValueError("Empty long-text fixture")

    def exact_length(length):
        return (text * ((length + len(text) - 1) // len(text)))[:length]

    for length in [255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 3073]:
        add(f"boundary-{length}", exact_length(length),
            "Repeated fixed text, exact token length; page size 256, prefill chunk 1024")
    common = exact_length(512)
    add("shared-prefix-english", common + fixtures[0]["prompt_token_ids"],
        "512 identical prefix tokens, English suffix")
    add("shared-prefix-chinese", common + fixtures[2]["prompt_token_ids"],
        "512 identical prefix tokens, Chinese chat suffix")
    manifest = dict(format_version=1, fixtures_source=str(args.fixtures.resolve()),
                    fixtures_sha256=digest(args.fixtures), cases=cases,
                    note="Additional suite; the original 64-token test remains a separate strict gate")
    write(suite / "cases.json", manifest)
    requests = [dict(id=c["id"], prompt_token_ids=c["prompt_token_ids"],
                     max_tokens=8, temperature=0.0, ignore_eos=True) for c in cases]
    write(suite / "workload.json", dict(name="attention-boundaries", seed=0, requests=requests))
    write(suite / "prefix-workload.json", dict(name="attention-prefix", seed=0,
                                               requests=requests[-2:]))
    for case in cases:
        write(suite / "tokens" / f"{case['id']}.json", case["prompt_token_ids"])
    print(f"Prepared {len(cases)} cases in {suite}", flush=True)


def hf(args):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM

    destination = new_directory(args.suite / "hf-eager")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager",
        local_files_only=True).cuda().eval()
    cases = read(args.suite / "cases.json")["cases"]
    arrays = {}
    with torch.inference_mode():
        for case in cases:
            ids = torch.tensor([case["prompt_token_ids"]], device="cuda")
            # Qwen3 supports logits_to_keep. Do not fallback to materializing
            # all-position vocabulary logits if the installed version does not.
            logits = model(input_ids=ids, use_cache=False, logits_to_keep=1).logits
            if logits.shape[1] != 1:
                raise ValueError("Expected only last-position logits from Qwen3")
            arrays[str(case["id"])] = logits[0, 0].float().cpu().numpy().copy()
            print(f"HF {case['name']}: {len(case['prompt_token_ids'])} tokens", flush=True)
    np.savez_compressed(destination / "logits.npz", **arrays)
    write(destination / "metadata.json", dict(engine="transformers-eager", dtype="bfloat16",
          model=str(args.model.resolve()), torch_version=torch.__version__,
          transformers_version=transformers.__version__,
          cases_sha256=digest(args.suite / "cases.json"),
          model_config_sha256=digest(args.model / "config.json")))


def invoke(command, log):
    print("Running:", " ".join(map(str, command)), flush=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false")
    with Path(log).open("w", encoding="utf-8") as stream:
        subprocess.run(list(map(str, command)), check=True, stdout=stream,
                       stderr=subprocess.STDOUT, env=env)


def rust(args):
    import numpy as np

    destination = new_directory(args.suite / args.label)
    cases = read(args.suite / "cases.json")["cases"]
    binary = args.binary.resolve()
    common = ["--model", args.model.resolve(), "--block-size", "256", "--num-blocks", "128",
              "--max-sequences", "4", "--max-model-len", "4096"]
    if args.attention_backend is not None:
        common.extend(["--attention-backend", args.attention_backend])
    metadata = dict(engine="nano-vllm-rs", dtype="bfloat16", label=args.label,
                    attention_backend=args.attention_backend or "binary-default",
                    binary=str(binary), binary_sha256=digest(binary),
                    model=str(args.model.resolve()),
                    model_config_sha256=digest(args.model / "config.json"),
                    cases_sha256=digest(args.suite / "cases.json"),
                    inherited_backend_env={k: v for k, v in os.environ.items()
                                           if k.startswith(("NANO_", "FLASH_", "CUDA_"))},
                    commands=[])
    # One process per case is slower to initialize, but keeps all model state
    # independent. This is correctness validation, never a timing benchmark.
    arrays = {}
    for case in cases:
        raw = destination / f"logits-{case['id']}.json"
        command = [binary, "logits", *common, "--eager", "--no-prefix-cache",
                   "--max-batch-tokens", "4096", "--tokens",
                   (args.suite / "tokens" / f"{case['id']}.json").resolve(), "--output", raw]
        metadata["commands"].append(list(map(str, command)))
        invoke(command, destination / f"logits-{case['id']}.log")
        result = read(raw)
        if result["prompt_token_ids"] != case["prompt_token_ids"]:
            raise ValueError("Native CLI returned a different teacher-forced context")
        arrays[str(case["id"])] = np.asarray(result["logits"], dtype=np.float32)
        # Losslessly preserve the numeric BF16 values as float32 and avoid
        # duplicating a large JSON representation in every artifact.
        raw.unlink()
    np.savez_compressed(destination / "logits.npz", **arrays)
    modes = [
        ("eager", "workload.json", ["--eager", "--no-prefix-cache"], "1024", "0"),
        ("graph", "workload.json", ["--no-prefix-cache"], "1024", "0"),
        ("chunked", "workload.json", ["--no-prefix-cache"], "64", "0"),
        ("prefix-cold", "prefix-workload.json", [], "1024", "0"),
        ("prefix-warm", "prefix-workload.json", ["--warm-prefix-cache"], "1024", "1"),
    ]
    for mode, workload, flags, chunk, warmups in modes:
        command = [binary, "bench", *common, *flags, "--max-batch-tokens", chunk,
                   "--workload", (args.suite / workload).resolve(), "--warmups", warmups,
                   "--repetitions", "1", "--output", destination / f"{mode}.json"]
        metadata["commands"].append(list(map(str, command)))
        invoke(command, destination / f"{mode}.log")
    write(destination / "metadata.json", metadata)


def comparison(reference, candidate):
    import numpy as np

    if reference.shape != candidate.shape or reference.ndim != 1:
        raise ValueError(f"Logit shape mismatch {reference.shape} / {candidate.shape}")
    finite = bool(np.isfinite(reference).all() and np.isfinite(candidate).all())
    if not finite:
        return dict(all_finite=False)
    left, right = reference.astype(np.float64), candidate.astype(np.float64)
    delta = right - left
    order = np.argsort(-left, kind="stable")[:2]
    ref_winner, other = map(int, order)
    winner = int(right.argmax())
    error = float(np.abs(delta).max())
    margin = float(left[ref_winner] - left[other])
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    return dict(all_finite=True, rmse=float(np.sqrt(np.mean(delta**2))), max_abs_error=error,
                cosine_similarity=float(np.dot(left, right) / denom) if denom else None,
                reference_top1=ref_winner, candidate_top1=winner,
                top1_matches=ref_winner == winner, reference_top1_margin=margin,
                reference_top1_logit=float(left[ref_winner]),
                candidate_reference_top1_logit=float(right[ref_winner]),
                candidate_top1_margin=float(np.sort(right)[-1] - np.sort(right)[-2]),
                reference_gap_to_candidate_winner=float(left[ref_winner] - left[winner]),
                reference_top1_protected_by_observed_linf=margin > 2 * error,
                bitwise_equal=bool(np.array_equal(reference, candidate)))


def outputs(path, expected):
    value = read(path)
    rows = value["runs"][0]["outputs"]
    result = {row["id"]: row["token_ids"] for row in rows}
    if len(result) != len(rows) or set(result) != set(expected):
        raise ValueError(f"Missing/duplicate/unexpected requests in {path}")
    if any(len(tokens) != 8 for tokens in result.values()):
        raise ValueError(f"Expected exactly 8 output tokens for every request: {path}")
    return result, value["runs"][0]["stats"]


def report(args):
    import numpy as np

    suite = args.suite
    manifest = read(suite / "cases.json")
    cases = manifest["cases"]
    labels = ["hf-eager", args.candidate]
    if args.baseline:
        labels.append(args.baseline)
    arrays, metadata = {}, {}
    for label in labels:
        arrays[label] = np.load(suite / label / "logits.npz", allow_pickle=False)
        metadata[label] = read(suite / label / "metadata.json")
        if metadata[label]["cases_sha256"] != digest(suite / "cases.json"):
            raise ValueError(f"Cases changed after recording {label}")
        if set(arrays[label].files) != {str(case["id"]) for case in cases}:
            raise ValueError(f"Missing or extra logits in {label}")
    if len({m["model_config_sha256"] for m in metadata.values()}) != 1:
        raise ValueError("Model config differs between validation backends")
    records = []
    for case in cases:
        key = str(case["id"])
        item = dict(id=case["id"], name=case["name"], prompt_tokens=len(case["prompt_token_ids"]),
                    candidate_vs_hf=comparison(arrays["hf-eager"][key], arrays[args.candidate][key]))
        if args.baseline:
            item["baseline_vs_hf"] = comparison(arrays["hf-eager"][key], arrays[args.baseline][key])
            item["candidate_vs_baseline"] = comparison(arrays[args.baseline][key], arrays[args.candidate][key])
        records.append(item)
    ids = [case["id"] for case in cases]
    mode_outputs, mode_stats = {}, {}
    for mode in ["eager", "graph", "chunked", "prefix-cold", "prefix-warm"]:
        expected = ids[-2:] if mode.startswith("prefix-") else ids
        mode_outputs[mode], mode_stats[mode] = outputs(suite / args.candidate / f"{mode}.json", expected)
    for item in records:
        index = item["id"]
        item["graph_matches_eager"] = mode_outputs["graph"][index] == mode_outputs["eager"][index]
        item["chunked_matches_eager"] = mode_outputs["chunked"][index] == mode_outputs["eager"][index]
        if index in ids[-2:]:
            item["prefix_modes_match_eager"] = all(mode_outputs[mode][index] == mode_outputs["eager"][index]
                                                   for mode in ["prefix-cold", "prefix-warm"])
    legacy = read(args.legacy_summary)
    required_legacy = ["all_first_tokens_match", "all_transformers_generations_match",
                       "graph_invariance", "chunk_invariance"]
    # Absence is a failure, not zero/default success; the path is deliberately
    # explicit so an old passing summary cannot silently be selected by default.
    legacy_pass = all(legacy.get(k) is True for k in required_legacy)
    finite = all(r["candidate_vs_hf"]["all_finite"] for r in records)
    invariance = all(r["graph_matches_eager"] and r["chunked_matches_eager"]
                     and r.get("prefix_modes_match_eager", True) for r in records)
    warm_used = mode_stats["prefix-warm"].get("cached_tokens", 0) > 0
    graph_decode = mode_stats["graph"].get("decode_steps", 0) > 1
    mismatches = [r["name"] for r in records if not r["candidate_vs_hf"].get("top1_matches", False)]
    result = dict(format_version=1, candidate=args.candidate, baseline=args.baseline,
                  metadata=metadata, cases=records,
                  legacy_strict_summary=str(args.legacy_summary.resolve()),
                  legacy_strict_summary_sha256=digest(args.legacy_summary),
                  legacy_strict_pass=legacy_pass,
                  legacy_nano_exact_generation_match=legacy.get("all_nano_generations_match"),
                  all_logits_finite=finite, exact_generation_mode_invariance=invariance,
                  prefix_warm_cache_exercised=warm_used, graph_mode_had_decode_steps=graph_decode,
                  graph_coverage="Graph enabled by CLI; decode steps verified; this CLI does not export graph counters or per-step logits",
                  additional_hf_top1_mismatches=mismatches,
                  diagnostic_review_required=bool(mismatches),
                  numerical_acceptance="No new numeric tolerance. A reference margin greater than twice the observed L-infinity error mathematically protects argmax; otherwise inspect the discrepancy. Susceptibility to rounding is not acceptance.",
                  mode_stats=mode_stats)
    result["strict_gates_pass"] = legacy_pass and finite and invariance and warm_used and graph_decode
    output = args.output or suite / f"{args.candidate}-validation.json"
    if output.exists():
        raise ValueError(f"Refusing to overwrite {output}; choose a new --output")
    write(output, result)
    print(json.dumps({key: result[key] for key in ["strict_gates_pass", "legacy_strict_pass",
                      "all_logits_finite", "exact_generation_mode_invariance",
                      "prefix_warm_cache_exercised", "additional_hf_top1_mismatches"]}, indent=2))
    print(f"Full diagnostics: {output}")
    if not result["strict_gates_pass"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["prepare", "hf", "rust", "report"]:
        command = commands.add_parser(name)
        command.add_argument("--suite", "--output-dir", dest="suite", type=Path, required=True)
        command.set_defaults(function=globals()[name])
        if name in ["hf", "rust"]:
            command.add_argument("--model", type=Path, default=Path("/root/huggingface/Qwen3-0.6B"))
        if name == "prepare":
            command.add_argument("--fixtures", type=Path,
                                 default=Path("benchmarks/optimization-results/transformers_reference.json"))
        if name == "rust":
            command.add_argument("--binary", type=Path, default=Path("target/release/nano-vllm-rs"))
            command.add_argument("--label", required=True)
            command.add_argument("--attention-backend", choices=["native", "flash"],
                                 help="Omit for an older binary without backend selection")
        if name == "report":
            command.add_argument("--candidate", required=True)
            command.add_argument("--baseline")
            command.add_argument("--legacy-summary", type=Path, required=True,
                                 help="This build's original 64-token strict result; never the old build's summary")
            command.add_argument("--output", type=Path)
    args = parser.parse_args()
    for key in ["label", "candidate", "baseline"]:
        value = getattr(args, key, None)
        if value is not None and (not value or not all(c.isalnum() or c in "-_" for c in value)
                                  or (key != "baseline" and value == "hf-eager")):
            parser.error(f"Invalid --{key}: choose a simple backend label")
    args.function(args)


if __name__ == "__main__":
    main()

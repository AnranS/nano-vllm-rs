"""Compare saved outputs; never substitutes synthetic results for missing data."""
import argparse
import json
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(path.read_text())


def output_map(value):
    return {entry.get("id", entry.get("request_id")): entry["token_ids"]
            for entry in value["runs"][0]["outputs"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("benchmarks/results"))
    args = parser.parse_args()
    reference = read(args.results / "transformers_reference.json")
    nano = {r["request_id"]: r for r in read(args.results / "nano-correctness.json")["results"]}
    eager = output_map(read(args.results / "rust-correctness-eager.json"))
    graph = output_map(read(args.results / "rust-correctness-graph.json"))
    chunked = output_map(read(args.results / "rust-correctness-chunked.json"))
    records = []
    for entry in reference["results"]:
        index = entry["request_id"]
        rust = read(args.results / f"rust-logits-{index}.json")
        left, right = np.array(entry["logits"]), np.array(rust["logits"])
        assert left.shape == right.shape
        delta = right - left
        nano_logits = np.array(nano[index]["logits"])
        nano_delta = right - nano_logits
        reference_tokens = entry["token_ids"]
        predicted = eager[index]
        records.append(dict(request_id=index, prompt_tokens=len(entry["prompt_token_ids"]),
            all_logits_finite=bool(np.isfinite(right).all()),
            max_abs_error=float(np.abs(delta).max()), rmse=float(np.sqrt(np.mean(delta**2))),
            cosine_similarity=float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))),
            reference_argmax=int(left.argmax()), rust_argmax=int(right.argmax()),
            first_token_matches=bool(left.argmax() == right.argmax()),
            reference_token_ids=reference_tokens, rust_token_ids=predicted,
            matching_generated_tokens=sum(a == b for a, b in zip(reference_tokens, predicted)),
            generated_tokens=len(predicted), exact_generation_match=reference_tokens == predicted,
            graph_matches_eager=graph[index] == predicted,
            chunked_matches_eager=chunked[index] == predicted,
            nano_token_ids=nano[index]["token_ids"],
            nano_exact_generation_match=nano[index]["token_ids"] == predicted,
            nano_max_abs_error=float(np.abs(nano_delta).max()),
            nano_rmse=float(np.sqrt(np.mean(nano_delta**2))),
            nano_cosine_similarity=float(np.dot(nano_logits, right) / (np.linalg.norm(nano_logits) * np.linalg.norm(right))),
            nano_argmax_matches=bool(nano_logits.argmax() == right.argmax())))
    report = dict(reference=reference["engine"], precision="bfloat16", cases=records,
        acceptance="Exact greedy generation against the independent Transformers fixture; graph/chunk invariance. Nano-vLLM numerical differences are reported separately, not asserted bitwise equal.",
        all_first_tokens_match=all(r["first_token_matches"] for r in records),
        all_transformers_generations_match=all(r["exact_generation_match"] for r in records),
        all_nano_generations_match=all(r["nano_exact_generation_match"] for r in records),
        graph_invariance=all(r["graph_matches_eager"] for r in records),
        chunk_invariance=all(r["chunked_matches_eager"] for r in records))
    (args.results / "correctness-summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not all((report["all_first_tokens_match"], report["all_transformers_generations_match"],
                report["graph_invariance"], report["chunk_invariance"])):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

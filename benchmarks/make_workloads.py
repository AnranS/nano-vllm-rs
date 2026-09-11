"""Generate shared, deterministic token workloads; no model execution."""
import argparse
import json
import random
from pathlib import Path


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def requests(prompts, lengths, temperature=0.0):
    return [dict(id=i, prompt_token_ids=p, max_tokens=n, temperature=temperature,
                 ignore_eos=True) for i, (p, n) in enumerate(zip(prompts, lengths))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/workloads"))
    args = parser.parse_args()
    for name, count, input_range, output_range in [
        ("latency-b1", 1, (128, 128), (128, 128)),
        ("throughput-b8", 8, (128, 128), (128, 128)),
        ("mixed-b32", 32, (100, 512), (128, 256)),
        ("nano-vllm-256", 256, (100, 1024), (100, 1024)),
    ]:
        rng = random.Random(0)
        prompts = [[rng.randint(0, 10000) for _ in range(rng.randint(*input_range))]
                   for _ in range(count)]
        lengths = [rng.randint(*output_range) for _ in range(count)]
        save(args.output / f"{name}.json", dict(name=name, seed=0,
             requests=requests(prompts, lengths, temperature=0.6)))
        print(name, "input", sum(map(len, prompts)), "output", sum(lengths))

    from transformers import AutoTokenizer
    save(args.output / "warmup-all-batches.json", dict(name="warmup-all-batches", seed=0,
        requests=requests([list(range(2, 34)) for _ in range(32)], list(range(2, 34)), temperature=0.6)))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    texts = ["The capital of France is", "What is 2 + 2? Answer briefly.",
             "用一句话介绍一下杭州。"]
    prompts = [tokenizer.encode(texts[0], add_special_tokens=False)]
    prompts.extend(tokenizer.encode(tokenizer.apply_chat_template([dict(role="user", content=text)],
        tokenize=False, add_generation_prompt=True, enable_thinking=False),
        add_special_tokens=False) for text in texts[1:])
    prompts.append(tokenizer.encode("The quick brown fox jumps over the lazy dog. " * 35,
                                    add_special_tokens=False))
    validation = dict(name="correctness", seed=0, requests=requests(prompts, [16] * len(prompts)))
    save(args.output / "correctness.json", validation)
    for index, prompt in enumerate(prompts):
        save(args.output / f"logits-{index}.json", prompt)
    # More than one full cache block is shared by all requests.
    common = tokenizer.encode("This is shared context for a cache test. " * 65,
                              add_special_tokens=False)
    cache_prompts = [common + tokenizer.encode(f" Question {i}: explain it.",
                     add_special_tokens=False) for i in range(8)]
    save(args.output / "prefix-b8.json", dict(name="prefix-b8", seed=0,
         requests=requests(cache_prompts, [64] * 8, temperature=0.6)))


if __name__ == "__main__":
    main()

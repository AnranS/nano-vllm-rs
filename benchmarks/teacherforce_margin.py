"""Tiny HF teacher-forced margin diagnostic. Run after timing benchmarks finish."""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def stats(logits):
    logits = logits.float().cpu()
    values, ids = logits.topk(10)
    return {
        "top10": [{"token_id": int(i), "logit": float(v)} for i, v in zip(ids, values)],
        "top1_minus_top2": float(values[0] - values[1]),
        "france_logit": float(logits[9625]),
        "germany_logit": float(logits[15344]),
        "france_minus_germany": float(logits[9625] - logits[15344]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/root/huggingface/Qwen3-0.6B")
    p.add_argument("--reference", default="/root/nano-vllm-rs/benchmarks/results/transformers_reference.json")
    p.add_argument("--output", default="/root/nano-vllm-rs/benchmarks/results/teacherforce-margin.json")
    args = p.parse_args()
    case = json.loads(Path(args.reference).read_text())["results"][0]
    prompt = case["prompt_token_ids"]
    forced = case["token_ids"][:5]
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
        attn_implementation="eager", local_files_only=True).cuda().eval()
    with torch.inference_mode():
        cache = None
        per_step = []
        for step in range(6):
            token_ids = prompt if step == 0 else [forced[step - 1]]
            result = model(torch.tensor([token_ids], device="cuda"),
                           past_key_values=cache, use_cache=True)
            cache = result.past_key_values
            per_step.append(stats(result.logits[0, -1]))
        full = model(torch.tensor([prompt + forced], device="cuda"), use_cache=False).logits[0, -1]
    output = {"precision": "bfloat16", "engine": "transformers-eager",
              "prompt_token_ids": prompt, "forced_token_ids": forced,
              "teacher_context": prompt + forced,
              "cached_steps": per_step, "full_prefill": stats(full),
              "note": "The sixth output decision is compared at an identical teacher-forced context; no new tolerance is introduced."}
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"cached_sixth": per_step[-1], "full_prefill": output["full_prefill"]}), flush=True)


if __name__ == "__main__":
    main()

"""Produce independent BF16 eager-Transformers logits and greedy generations."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import transformers
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
        attn_implementation="eager", local_files_only=True).cuda().eval()
    workload = json.loads(args.workload.read_text())
    results = []
    with torch.inference_mode():
        for request in workload["requests"]:
            ids = torch.tensor([request["prompt_token_ids"]], device="cuda")
            first = model(input_ids=ids, use_cache=False).logits[0, -1].float().cpu().tolist()
            output = model.generate(ids, max_new_tokens=request["max_tokens"], do_sample=False,
                eos_token_id=None, pad_token_id=151643, use_cache=True)
            generated = output[0, ids.shape[1]:].cpu().tolist()
            results.append(dict(request_id=request["id"], prompt_token_ids=request["prompt_token_ids"],
                logits=first, token_ids=generated))
            print("reference request", request["id"], "generated", generated, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(engine="transformers-eager", precision="bfloat16",
        torch_version=torch.__version__, transformers_version=transformers.__version__, results=results)))


if __name__ == "__main__":
    main()

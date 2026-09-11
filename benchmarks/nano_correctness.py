"""Check nano-vLLM's actual compiled model with an explicit greedy sampler.

The original public SamplingParams rejects temperature=0, so this diagnostic
replaces only sampling with argmax. It is not used for performance results.
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("/root/nano-vllm"))
    parser.add_argument("--max-batch-tokens", type=int, default=1024)
    parser.add_argument("--batched", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.reference))
    import torch
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.model_runner import ModelRunner
    from nanovllm.engine.scheduler import Scheduler

    def allocate(runner):
        cfg, hf = runner.config, runner.config.hf_config
        cfg.num_kvcache_blocks = 32
        dim = getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)
        runner.kv_cache = torch.empty(2, hf.num_hidden_layers, 32,
            cfg.kvcache_block_size, hf.num_key_value_heads, dim)
        layer = 0
        for module in runner.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache, module.v_cache = runner.kv_cache[0, layer], runner.kv_cache[1, layer]
                layer += 1

    ModelRunner.allocate_kv_cache = allocate
    llm = LLM(str(args.model), enforce_eager=True, max_model_len=4096,
              max_num_seqs=4, max_num_batched_tokens=args.max_batch_tokens)

    class Greedy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first_logits = None

        def forward(self, logits, temperatures):
            if self.first_logits is None:
                self.first_logits = logits[0].float().cpu().tolist()
            return logits.argmax(dim=-1)

    sampler = Greedy()
    llm.model_runner.sampler = sampler
    results = []
    requests = json.loads(args.workload.read_text())["requests"]
    if args.batched:
        outputs = llm.generate([r["prompt_token_ids"] for r in requests],
            [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=r["max_tokens"])
             for r in requests], use_tqdm=False)
        results = [dict(request_id=r["id"], token_ids=o["token_ids"])
                   for r, o in zip(requests, outputs)]
        args.output.write_text(json.dumps(dict(engine="nano-vllm-argmax-diagnostic",
            max_batch_tokens=args.max_batch_tokens, results=results)))
        print(json.dumps(results), flush=True)
        return
    for request in requests:
        llm.scheduler = Scheduler(llm.model_runner.config)
        sampler.first_logits = None
        output = llm.generate([request["prompt_token_ids"]], SamplingParams(
            temperature=0.6, ignore_eos=True, max_tokens=request["max_tokens"]), use_tqdm=False)[0]
        results.append(dict(request_id=request["id"], logits=sampler.first_logits,
                            token_ids=output["token_ids"]))
        print("nano reference", request["id"], output["token_ids"], flush=True)
    args.output.write_text(json.dumps(dict(engine="nano-vllm-argmax-diagnostic", results=results)))


if __name__ == "__main__":
    main()

"""Serial before/after/Python measurements with the same saved workload and KV budget."""
import argparse
import json
import os
from pathlib import Path
import statistics
from run_suite import run_monitored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profiles', nargs='+', default=['latency-b1', 'throughput-b8', 'mixed-b32'])
    parser.add_argument('--engines', nargs='+', choices=['before', 'after', 'python'], default=['before', 'after', 'python'])
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--warmup-workload', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('benchmarks/optimization-results'))
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error('repetitions must be positive')
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    results = args.output_dir.resolve()
    logs = results / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
        TORCHINDUCTOR_CACHE_DIR=str(root / '.cache/torchinductor'),
        TRITON_CACHE_DIR=str(root / '.cache/triton'))
    commands = {
        'before': [str(root / '.optimization/baseline/nano-vllm-rs'), 'bench'],
        'after': [str(root / 'target/release/nano-vllm-rs'), 'bench'],
        'python': ['/root/nano-vllm/.venv/bin/python', 'benchmarks/python_reference.py'],
    }
    for profile in args.profiles:
        workload = 'prefix-b8' if profile.startswith('prefix-b8-') else profile
        for engine in args.engines:
            name = f'{engine}-{profile}'
            command = commands[engine] + ['--model', '/root/huggingface/Qwen3-0.6B',
                '--workload', f'benchmarks/workloads/{workload}.json',
                '--output', str(results / f'{name}.json'), '--warmups', '1',
                '--repetitions', str(args.repetitions), '--block-size', '256', '--num-blocks', '128',
                '--max-sequences', '32', '--max-model-len', '4096', '--max-batch-tokens', '1024']
            if profile.endswith('-warm'):
                command += ['--warm-prefix-cache']
            if args.warmup_workload:
                command += ['--warmup-workload', str(args.warmup_workload)]
            print(f'START {name}', flush=True)
            run_monitored(command, logs / f'{name}.log', env)
            data = json.loads((results / f'{name}.json').read_text())
            rates = [run['stats']['output_tokens_per_second'] for run in data['runs']]
            print(f'DONE {name}: median {statistics.median(rates):.2f} output tok/s; runs {rates}', flush=True)


if __name__ == '__main__':
    main()

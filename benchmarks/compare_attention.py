"""Serial round-one/FlashAttention/Python comparison using identical saved workloads."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import statistics


def absolute_path(value):
    # Do not resolve symlinks: resolving .venv/bin/python can bypass its venv.
    return Path(os.path.abspath(Path(value).expanduser()))


def executable_path(value):
    candidate = str(Path(value).expanduser())
    if os.sep not in candidate:
        candidate = shutil.which(candidate) or candidate
    return absolute_path(candidate)


def main():
    root = Path(__file__).resolve().parents[1]
    baseline = root / 'benchmarks/baselines/round1'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profiles', nargs='+', default=['latency-b1', 'throughput-b8', 'mixed-b32'])
    parser.add_argument('--engines', nargs='+', choices=['round1', 'flash', 'python'], default=['round1', 'flash', 'python'])
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--warmup-workload', type=Path)
    parser.add_argument('--output-dir', type=Path, default=root / 'benchmarks/attention-results')
    parser.add_argument('--model', type=Path, default=Path('/root/huggingface/Qwen3-0.6B'),
                        help='Local Qwen3 model directory')
    parser.add_argument('--python', default='/root/nano-vllm/.venv/bin/python',
                        help='Python interpreter containing the reference dependencies; path or executable name')
    parser.add_argument('--reference', type=Path, default=Path('/root/nano-vllm'),
                        help='Python nano-vLLM source checkout')
    parser.add_argument('--binary', type=Path, default=root / 'target/release/nano-vllm-rs',
                        help='Current Rust executable with --attention-backend flash support')
    parser.add_argument('--baseline-binary', type=Path,
                        default=baseline / 'target/release/nano-vllm-rs',
                        help='Round-one executable; defaults to the build from the included historical sources')
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error('repetitions must be positive')

    # Resolve user-relative arguments before switching to the repository directory.
    args.model = absolute_path(args.model)
    args.reference = absolute_path(args.reference)
    args.binary = absolute_path(args.binary)
    args.baseline_binary = absolute_path(args.baseline_binary)
    args.python = executable_path(args.python)
    results = absolute_path(args.output_dir)
    if args.warmup_workload:
        args.warmup_workload = absolute_path(args.warmup_workload)
        if not args.warmup_workload.is_file():
            parser.error(f'warmup workload does not exist: {args.warmup_workload}')

    if 'round1' in args.engines and not (
        args.baseline_binary.is_file() and os.access(args.baseline_binary, os.X_OK)
    ):
        build = shlex.join([
            'cargo', 'build', '--release', '--locked', '--manifest-path',
            str(baseline / 'Cargo.toml'), '--target-dir', str(baseline / 'target'),
        ])
        parser.error(
            f'round-one baseline executable is missing or not executable: {args.baseline_binary}\n'
            f'Build the included historical sources first:\n  {build}\n'
            'Or pass --baseline-binary /path/to/your/round1/nano-vllm-rs.'
        )
    if 'flash' in args.engines and not (args.binary.is_file() and os.access(args.binary, os.X_OK)):
        parser.error(f'current Rust executable is missing or not executable: {args.binary}; '
                     'run cargo build --release --locked, or pass --binary')
    if 'python' in args.engines:
        if not (args.python.is_file() and os.access(args.python, os.X_OK)):
            parser.error(f'Python executable does not exist or is not executable: {args.python}')
        if not (args.reference / 'nanovllm').is_dir():
            parser.error(f'Python reference checkout has no nanovllm package: {args.reference}')
    if not (args.model / 'config.json').is_file():
        parser.error(f'local model directory has no config.json: {args.model}')
    for profile in args.profiles:
        workload = 'prefix-b8' if profile.startswith('prefix-b8-') else profile
        if not (root / f'benchmarks/workloads/{workload}.json').is_file():
            parser.error(f'saved workload does not exist: benchmarks/workloads/{workload}.json; '
                         'run python3 benchmarks/make_workloads.py')

    from run_suite import run_monitored

    os.chdir(root)
    logs = results / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
               TORCHINDUCTOR_CACHE_DIR=str(root / '.cache/torchinductor'),
               TRITON_CACHE_DIR=str(root / '.cache/triton'))
    commands = {
        'round1': [str(args.baseline_binary), 'bench'],
        'flash': [str(args.binary), 'bench', '--attention-backend', 'flash'],
        'python': [str(args.python), str(root / 'benchmarks/python_reference.py'),
                   '--reference', str(args.reference)],
    }
    for profile in args.profiles:
        workload = 'prefix-b8' if profile.startswith('prefix-b8-') else profile
        for engine in args.engines:
            name = f'{engine}-{profile}'
            command = commands[engine] + ['--model', str(args.model),
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

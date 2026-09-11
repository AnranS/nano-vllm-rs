"""Validate every measured request and summarize the optional Flash backend."""
from pathlib import Path
import csv
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'benchmarks/attention-results'
PROFILES = ['latency-b1', 'throughput-b8', 'mixed-b32', 'nano-vllm-256', 'prefix-b8-cold', 'prefix-b8-warm']
ENGINES = ['round1', 'flash', 'python']

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def summarize():
    records = []
    validated = []
    for profile in PROFILES:
        workload_name = 'prefix-b8' if profile.startswith('prefix-b8-') else profile
        workload = json.loads((ROOT / f'benchmarks/workloads/{workload_name}.json').read_text())
        requests = {r['id']: r for r in workload['requests']}
        rates = {}
        for engine in ENGINES:
            path = RESULTS / f'{engine}-{profile}.json'
            data = json.loads(path.read_text())
            runs = data['runs']
            assert len(runs) == (1 if profile == 'nano-vllm-256' else 3), path
            for run in runs:
                ids = []
                for output in run['outputs']:
                    identifier = output.get('id', output.get('request_id'))
                    ids.append(identifier)
                    assert len(output['token_ids']) == requests[identifier]['max_tokens'], path
                    assert 0 <= output['ttft_ms'] <= output['latency_ms'], path
                assert len(ids) == len(requests) and set(ids) == set(requests), path
                assert run['stats']['input_tokens'] == sum(len(r['prompt_token_ids']) for r in requests.values()), path
                assert run['stats']['output_tokens'] == sum(r['max_tokens'] for r in requests.values()), path
            values = [r['stats']['output_tokens_per_second'] for r in runs]
            rates[engine] = statistics.median(values)
            record = dict(profile=profile, engine=engine, runs=len(runs),
                          median_tokens_s=rates[engine], min_tokens_s=min(values), max_tokens_s=max(values))
            records.append(record)
            monitor = json.loads((RESULTS/f'logs/{engine}-{profile}.monitor.json').read_text())
            assert monitor['returncode'] == 0, path
            validated.append(dict(file=path.name, sha256=digest(path), requests=len(requests), runs=len(runs)))
        for record in records[-3:]:
            record['flash_over_round1'] = rates['flash'] / rates['round1']
            record['flash_over_python'] = rates['flash'] / rates['python']
    with (RESULTS/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    metadata = dict(checks='Exact request IDs/output counts, total tokens, latency ordering, process exit status. Numerical correctness is separately recorded, not implied by performance.',
                    files=validated,
                    round1_binary_sha256=digest(ROOT/'.optimization/round1/nano-vllm-rs'),
                    flash_binary_sha256=digest(ROOT/'target/release/nano-vllm-rs'),
                    correctness_summary='flash/correctness-summary.json',
                    validation_report='../attention-validation/flash-final-validation.json')
    (RESULTS/'delivery-validation.json').write_text(json.dumps(metadata, indent=2))
    lines = ['| 负载 | 上轮 Rust | 本轮 Flash Rust | Python 原版 | 相对上轮 | 相对 Python |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for profile in PROFILES:
        row = {r['engine']: r for r in records if r['profile'] == profile}
        r = row['flash']
        lines.append(f"| {profile} | {row['round1']['median_tokens_s']:,.1f} | {r['median_tokens_s']:,.1f} | {row['python']['median_tokens_s']:,.1f} | {r['flash_over_round1']:.2f}× | {r['flash_over_python']:.2f}× |")
    (RESULTS/'performance-table.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
    return records

if __name__ == '__main__':
    summarize()

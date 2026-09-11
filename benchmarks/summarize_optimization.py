"""Summarize measured optimization results; profiler timings are kept separate."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'benchmarks/optimization-results'
PROFILES = {
    'latency-b1': '单请求 128→128',
    'throughput-b8': '8 请求 128→128',
    'mixed-b32': '32 请求混合长度',
    'nano-vllm-256': '256 请求混合长度',
    'prefix-b8-cold': '共享前缀冷缓存',
    'prefix-b8-warm': '共享前缀热缓存',
}


def read(path):
    return json.loads(path.read_text())


def profiler_tables(path):
    lines = path.read_text().splitlines()
    tables = []
    for index, line in enumerate(lines):
        if line.startswith('Time (%),'):
            end = index + 1
            while end < len(lines) and lines[end].strip():
                end += 1
            tables.append(list(csv.DictReader(lines[index:end])))
    return tables


def main():
    lines = ['# nano-vllm-rs 性能优化实测', '',
        f'生成时间：{datetime.now(timezone.utc).isoformat()}。WSL2：`llm-ubuntu-24.04`；项目：`/root/nano-vllm-rs`。', '',
        '模型为 Qwen3-0.6B BF16，GPU 为 RTX 5070 Ti 16GB。所有吞吐测试关闭 profiler，两版 Rust 与 Python 串行运行。', '',
        '## 条件与改动', '',
        '- 相同 token 输入、固定输出长度、temperature=0.6、ignore_eos=true；KV 128×256 块（3.5 GiB）、max_sequences=32、单次 token 预算 1024、上下文上限 4096。',
        '- Rust 前后使用同一概率采样定义和 seed 推导；Python 随机数实现不同。模型加载、分词、输出解码和预热不计入吞吐。',
        '- 小负载完整预热 1 次、测量 3 次取中位数；完整 256 请求使用覆盖 decode batch 32..1 的短预热、测量 1 次。',
        '- Python 的 max_sequences 约束每轮调度数，Rust 约束总 active 数；Python 对照依然是相同配置值下的实现比较，不是严格等价的调度策略比较。',
        '- Attention 使用常见 head_dim 特化、寄存器保存 query、按物理 KV 块查表及成组预取，并保持原有 V 累加顺序和运行时缩放系数；保留通用维度路径，未更换计算精度或 KV 预算。',
        '- 温度采样改为词表分区归约；RoPE 三角函数在加载时缓存；每轮 8 次元数据上传合并为 1 次，设备缓冲区视图保留分配与 Graph 生命周期。',
        '- 调度按可用 KV 容量接纳新请求，减少新 prefill 对已有 decode 请求的驱逐；保持 FIFO 接纳及并发上限。', '',
        '## 输出吞吐', '',
        '单位为输出 token/s，分母包含 prefill、decode、调度、数据传输和同步。', '',
        '| 负载 | 测量次数 | Rust 优化前 | Rust 优化后 | 加速比 | Python nano-vLLM | 优化后/Python |',
        '|---|---:|---:|---:|---:|---:|---:|']
    rows = []
    data = {}
    for profile, label in PROFILES.items():
        paths = {engine: RESULTS / f'{engine}-{profile}.json' for engine in ['before', 'after', 'python']}
        if not all(path.exists() for path in paths.values()):
            raise SystemExit(f'Missing completed comparison for {profile}')
        group = {engine: read(path) for engine, path in paths.items()}
        counts = []
        rates = {}
        for engine, value in group.items():
            totals = []
            for run in value['runs']:
                total = sum(len(output['token_ids']) for output in run['outputs'])
                assert total == run['stats']['output_tokens']
                totals.append((run['stats']['input_tokens'], total, len(run['outputs'])))
                assert abs(total / run['stats']['elapsed_seconds'] - run['stats']['output_tokens_per_second']) < 1e-6
            assert len(set(totals)) == 1
            counts.append(totals[0])
            rates[engine] = statistics.median(run['stats']['output_tokens_per_second'] for run in value['runs'])
            rows.append(dict(engine=engine, profile=profile, repetitions=len(value['runs']),
                input_tokens=totals[0][0], output_tokens=totals[0][1],
                throughput_median=rates[engine],
                throughput_min=min(run['stats']['output_tokens_per_second'] for run in value['runs']),
                throughput_max=max(run['stats']['output_tokens_per_second'] for run in value['runs']),
                elapsed_median=statistics.median(run['stats']['elapsed_seconds'] for run in value['runs'])))
        assert len(set(counts)) == 1
        repeats = len(group['after']['runs'])
        assert all(len(value['runs']) == repeats for value in group.values())
        lines.append(f"| {label} | {repeats} | {rates['before']:,.2f} | {rates['after']:,.2f} | {rates['after']/rates['before']:.2f}× | {rates['python']:,.2f} | {rates['after']/rates['python']:.1%} |")
        data[profile] = group
    full = data['nano-vllm-256']
    lines += ['', '## 完整负载调度', '', '| 指标 | Rust 优化前 | Rust 优化后 | Python |', '|---|---:|---:|---:|']
    for key, label in [('prefill_tokens', '实际 prefill token（含重算）'), ('decode_steps', 'Decode 调度轮数'), ('preemptions', '抢占次数')]:
        values = [full[engine]['runs'][0]['stats'].get(key) for engine in ['before', 'after', 'python']]
        lines.append('| ' + label + ' | ' + ' | '.join(f'{v:,}' if v is not None else '未记录' for v in values) + ' |')
    correctness = read(RESULTS / 'correctness-summary.json')
    assert all(correctness[key] for key in ['all_transformers_generations_match', 'all_first_tokens_match', 'chunk_invariance', 'graph_invariance'])
    assert all(case['all_logits_finite'] for case in correctness['cases'])
    before_gpu, before_api = profiler_tables(RESULTS / 'profiles/baseline-b8-stats.csv')
    after_gpu, after_api = profiler_tables(RESULTS / 'profiles/optimized-b8-stats.csv')
    mixed_gpu = profiler_tables(RESULTS / 'profiles/baseline-mixed32-stats.csv')[0]
    def kernel_ms(table, name):
        return sum(float(row['Total Time (ns)']) for row in table if name in row['Name']) / 1e6
    def kernel_percent(table, name):
        return 100 * kernel_ms(table, name) / sum(float(row['Total Time (ns)']) / 1e6 for row in table)
    def calls(table, name):
        return sum(int(row['Num Calls']) for row in table if row['Name'] == name)
    lines += ['', '## 正确性', '',
        f"- 独立 Transformers 验证：4 组输入共 64 个贪心输出 token，全序列一致：{correctness['all_transformers_generations_match']}；首 token 一致：{correctness['all_first_tokens_match']}。",
        f"- 完整 prefill / 分块 prefill 一致：{correctness['chunk_invariance']}；eager / CUDA Graph 一致：{correctness['graph_invariance']}。",
        '- 22 个普通测试、6 个真实 GPU 测试及 Clippy 检查通过。覆盖调度、缓存 RoPE、分区采样及缓冲区生命周期，日志保存在 optimization-results/。',
        '- 实验中的并行 V 归约未通过原有生成一致性检查，因此最终采用保持累加顺序的版本；没有放宽验收标准。', '',
        '## Profiler 证据与局限', '',
        '- Nsight Systems 仅捕获预热后的测量范围，并启用 CUDA Graph node 跟踪。其耗时会受追踪开销影响，因此不混入上表。',
        f"- 优化前的 8 请求分析中，attention 约占 GPU 内核执行时间的 {kernel_percent(before_gpu, 'attention'):.1f}%；32 请求混合长度下约占 {kernel_percent(mixed_gpu, 'attention'):.1f}%。两版的矩阵乘内核与调用次数基本相同，优化重点据此放在 attention。",
        f"- 在 8 请求的本次追踪中，attention 内核累计执行时间由 {kernel_ms(before_gpu, 'attention'):.2f} ms 降至 {kernel_ms(after_gpu, 'attention'):.2f} ms，采样由 {kernel_ms(before_gpu, 'sample'):.2f} ms 降至 {kernel_ms(after_gpu, 'sample'):.2f} ms（分区与最终归约合计）。这些是 profiler 观测值，不是端到端加速倍数。",
        f"- 同一 128 步负载的 cudaMemcpyAsync 调用从 {calls(before_api, 'cudaMemcpyAsync')} 次降至 {calls(after_api, 'cudaMemcpyAsync')} 次，cudaStreamSynchronize 从 {calls(before_api, 'cudaStreamSynchronize')} 次降至 {calls(after_api, 'cudaStreamSynchronize')} 次。Memcpy API 时长还包含等待 GPU 的时间，不能当作纯数据拷贝耗时。",
        '- 原始 profiler 汇总保存在 optimization-results/profiles/；完整 .nsys-rep 与 SQLite 在 WSL 项目的 .optimization/profiles/。',
        '- 显存日志是每 0.5 秒整卡采样，包含桌面和其他应用变化；不能当作进程分配器的精确峰值。单次完整负载也不能说明统计稳定性。',
        '- 当前仍为单卡 dense Qwen3、4K 上下文版本；没有张量并行、量化或 HTTP 流式服务。', '',
        '## 复现', '', '```bash', 'cd /root/nano-vllm-rs', 'source scripts/env.sh', 'cargo build --release --locked --offline',
        'python3 benchmarks/compare_optimization.py --profiles latency-b1 throughput-b8 mixed-b32',
        'python3 benchmarks/compare_optimization.py --profiles nano-vllm-256 --repetitions 1 --warmup-workload benchmarks/workloads/warmup-all-batches.json',
        'python3 benchmarks/compare_optimization.py --profiles prefix-b8-cold prefix-b8-warm',
        'python3 benchmarks/summarize_optimization.py', '```', '',
        'before 使用本机保留的 .optimization/baseline/nano-vllm-rs；after 使用当前 release 可执行程序。对照脚本、保存的工作负载与逐轮 JSON 一同交付。', '']
    (ROOT / 'OPTIMIZATION.md').write_text('\n'.join(lines), encoding='utf-8')
    with (RESULTS / 'summary.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    sources = [*ROOT.glob('src/**/*.rs'), *ROOT.glob('kernels/*.cu'), *ROOT.glob('benchmarks/*.py'), ROOT/'Cargo.lock', ROOT/'Cargo.toml', ROOT/'build.rs']
    manifest = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(sources)}
    (RESULTS / 'source-sha256.json').write_text(json.dumps(manifest, indent=2))
    print(ROOT / 'OPTIMIZATION.md')


if __name__ == '__main__':
    main()

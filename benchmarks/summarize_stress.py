"""Build a report from completed stress suites, verifying their raw gzip evidence."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def number(value, digits=1):
    return '不可用' if value is None else f'{value:,.{digits}f}'


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suites', nargs='+', type=Path, default=[
        root/'benchmarks/stress-results/flash-10', root/'benchmarks/stress-results/reference-3'])
    parser.add_argument('--output', type=Path, default=root/'STRESS_TEST.md')
    args = parser.parse_args()
    suites = [path.resolve() for path in args.suites]
    records, contexts, evidence = {}, [], []
    for directory in suites:
        suite = json.loads((directory/'summary.json').read_text())
        if suite['status'] not in ['passed', 'failed']:
            raise ValueError(f'Suite is incomplete: {directory}')
        contexts.append(suite)
        for entry in suite['backends']:
            backend = entry['backend']
            if entry['valid_repetitions'] != suite['requested_repetitions'] or entry['completed_repetitions'] != suite['requested_repetitions']:
                raise ValueError(f'Cannot present a complete performance comparison for invalid or missing runs: {backend}; inspect {directory}/summary.json')
            if backend in records:
                raise ValueError(f'Duplicate backend: {backend}')
            archive = entry['raw_result_archive']
            path = directory/Path(archive['compressed_path']).name
            if sha256(path) != archive['compressed_sha256']:
                raise ValueError(f'Archive changed: {path}')
            digest = hashlib.sha256()
            with gzip.open(path, 'rb') as stream:
                for block in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(block)
            if digest.hexdigest() != archive['uncompressed_sha256']:
                raise ValueError(f'Raw result checksum differs: {path}')
            records[backend] = entry
            evidence.append(dict(backend=backend, path=str(path.relative_to(root)),
                                 gzip_sha256=archive['compressed_sha256'],
                                 raw_sha256=archive['uncompressed_sha256']))
    if set(records) != {'native', 'flash', 'python'}:
        raise ValueError('Report requires native, flash, and Python results')
    for key in ['workload_sha256', 'warmup_workload_sha256', 'model_config_sha256', 'seed', 'settings']:
        if any(c[key] != contexts[0][key] for c in contexts[1:]):
            raise ValueError(f'Incomparable suite setting: {key}')
    expected_settings = dict(warmups=1, block_size=256, num_blocks=128,
                             max_sequences=32, max_model_len=4096, max_batch_tokens=1024)
    if contexts[0]['settings'] != expected_settings or contexts[0]['requests_per_repetition'] != 256:
        raise ValueError('This publication report describes the recorded 256-request configuration; adapt its narrative before using other settings')
    if len({c['rust_binary']['sha256'] for c in contexts}) != 1:
        raise ValueError('Native and Flash were not measured from the same executable')
    labels = {'native': 'Rust native', 'flash': 'Rust Flash', 'python': 'Python nano-vLLM'}
    order = ['native', 'flash', 'python']
    def metric(engine, name, statistic='median'):
        return records[engine]['distributions_across_valid_repetitions'][name][statistic]
    native, flash, python = [metric(e, 'output_tokens_per_second') for e in order]
    ctx = contexts[0]
    rows = []
    for backend in order:
        entry = records[backend]
        for run in entry['runs']:
            rows.append(dict(backend=backend, **{key: run.get(key) for key in [
                'repetition', 'valid', 'output_tokens', 'elapsed_seconds', 'output_tokens_per_second',
                'ttft_p50_ms', 'ttft_p95_ms', 'inter_token_p50_ms', 'inter_token_p95_ms',
                'prefill_tokens', 'preemptions', 'decode_steps', 'generated_tokens_sha256']}))
    result_dir = root/'benchmarks/stress-results'
    result_dir.mkdir(parents=True, exist_ok=True)
    with (result_dir/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (result_dir/'evidence.json').write_text(json.dumps(dict(raw_results=evidence,
        binary_sha256=ctx['rust_binary']['sha256'], workload_sha256=ctx['workload_sha256'],
        source_commit=ctx['build_identity']['commit'],
        all_operational_stress_checks_pass=all(r['passed'] for r in records.values())), indent=2)+'\n')
    passed = all(r['passed'] for r in records.values())
    lines = ['# 连续压测与 Python 对比报告', '',
        f"本次使用同一 Rust release 二进制，Flash 连续运行 **{records['flash']['completed_repetitions']} 轮**，native运行 **{records['native']['completed_repetitions']} 轮**，Python运行 **{records['python']['completed_repetitions']} 轮**；每个后端仅加载模型一次。每轮提交 {ctx['requests_per_repetition']} 个请求、输入 {ctx['expected_input_tokens_per_repetition']:,} token、固定生成 {ctx['expected_output_tokens_per_repetition']:,} token。", '',
        f"Flash 吞吐中位数为 **{flash:,.1f} output token/s**，native 为 **{native:,.1f}**，Python 为 **{python:,.1f}**。Flash 为 native 的 **{flash/native:.2f}×**，相对 Python **{(flash/python-1)*100:+.1f}%**。", '',
        f"运行完整性与跨轮输出校验：**{'通过' if passed else '存在失败，见原始 summary'}**。此结果不替代模型数值验收：Flash 的 chunk64 生成仍有已记录的分叉，默认保持 native，详见 [ATTENTION.md](ATTENTION.md)。", '',
        '## 条件与计时口径', '',
        '2026-09-11，本机 WSL2 Ubuntu 24.04、RTX 5070 Ti 16GB、Qwen3-0.6B、BF16。Rust 1.98.1 / CUDA Toolkit 13.2 / 驱动 610.88；Python参考使用 torch 2.12.1+cu130、transformers 5.15.1、flash_attn 2.8.3.post1。', '',
        '共同参数：block_size=256、num_blocks=128（KV 3.5 GiB）、max_sequences=32、max_model_len=4096、max_batch_tokens=1024；Graph开启、temperature=0.6、ignore_eos=true、固定seed与请求次序。每轮清空可复用前缀元数据，保留模型与Graph。使用 warmup-all-batches.json 单独预热一次。', '',
        '三个后端串行执行，正式计时期间没有并行GPU测试、Nsight或编译任务。每个后端的重复轮次都在同一模型进程内完成；模型加载、JSON写出与单独预热不计入吞吐，prefill、decode、调度、采样、传输和同步计入。当前是CLI队列负载，未测HTTP/QPS服务。', '',
        'Rust限制总active请求数，Python限制每轮调度数，因此相同max_sequences并不代表完全相同的调度过程。TTFT从整批提交时开始，包含排队。TTFT/ITL先在每轮内求p50/p95，再对轮次取中位数；Python ITL采用其从真实间隔计算的stats分位数，没有用mean_tpot伪造分位数。', '',
        '## 吞吐与延迟', '',
        '| 后端 | 轮数 | 计时合计 s | 吞吐中位数 token/s | 吞吐范围 | CV | TTFT p50 / p95 ms | ITL p50 / p95 ms |',
        '| --- | ---: | ---: | ---: | --- | ---: | --- | --- |']
    for e in order:
        v = records[e]
        lines.append(f"| {labels[e]} | {v['completed_repetitions']} | {v['measured_seconds']:.1f} | {metric(e,'output_tokens_per_second'):,.1f} | {metric(e,'output_tokens_per_second','min'):,.1f}–{metric(e,'output_tokens_per_second','max'):,.1f} | {metric(e,'output_tokens_per_second','coefficient_of_variation_percent'):.2f}% | {metric(e,'ttft_p50_ms'):,.1f} / {metric(e,'ttft_p95_ms'):,.1f} | {metric(e,'inter_token_p50_ms'):.3f} / {metric(e,'inter_token_p95_ms'):.3f} |")
    lines += ['', 'CV为轮次吞吐的总体标准差/均值。连续短测的波动不等于跨机器或长时间稳定性保证；本次没有构造统计显著性结论。', '',
        '### 每轮吞吐', '', '| 轮次 | Rust native | Rust Flash | Python |', '| ---: | ---: | ---: | ---: |']
    for i in range(max(len(records[e]['runs']) for e in order)):
        values = [number(records[e]['runs'][i]['output_tokens_per_second']) if i < len(records[e]['runs']) else '—' for e in order]
        lines.append(f"| {i+1} | {' | '.join(values)} |")
    lines += ['', '## 运行完整性与确定性', '',
        '| 后端 | 完成轮次 / 请求 | 输出总 token | 进程退出码 | 同后端跨轮输出 SHA256 |',
        '| --- | --- | ---: | ---: | --- |']
    for e in order:
        v=records[e]
        lines.append(f"| {labels[e]} | {v['completed_repetitions']} / {v['completed_repetitions']*ctx['requests_per_repetition']:,} | {sum(x['output_tokens'] for x in v['runs']):,} | {v['process_returncode']} | {'一致' if v['repeated_output_checksum_consistent'] else '不一致'} |")
    lines += ['', '校验包含每请求ID唯一且齐全、每请求输出长度、输入/输出总token、有限非负延迟，以及 output_tokens/elapsed_seconds 与记录吞吐相符。Rust额外校验逐请求ITL数量，Python核验其从实际间隔计算的stats分位数有限有效。输出SHA按ID排序后只哈希token，排除时间。Rust和Python的随机采样实现不同，未要求跨后端token相同。', '',
        '默认native的旧64-token/HF、Graph、chunk64回归通过。Flash的旧eager64-token/HF与Graph通过，chunk64失败；扩展20个固定上下文top1均与HF一致，Graph20/20、chunk18/20。既有严格失败保持原样，没有为了发布修改门槛。', '',
        '## 显存采样', '',
        '| 后端 | 整卡峰值 MiB | 末1/4采样中位 MiB | 末1/4范围 MiB | 后1/2线性斜率 MiB/min |',
        '| --- | ---: | ---: | ---: | ---: |']
    for e in order:
        m=records[e]['gpu_memory_trend']
        lines.append(f"| {labels[e]} | {number(m.get('peak_mib'))} | {number(m.get('last_quarter_median_mib'))} | {number(m.get('last_quarter_range_mib'))} | {number(m.get('latter_half_linear_slope_mib_per_minute'),3)} |")
    lines += ['', '这些是约0.5秒一次的nvidia-smi整卡采样，包含其他应用、模型启动、预热和退出边界，不是每轮分配器精确峰值。斜率只是描述统计；即使近零，也不能据此宣布无内存泄漏。本次覆盖同进程重复推理，不覆盖反复加载/销毁模型或Graph的长期寿命测试。', '',
        '## 复现与证据', '',
        f"测量时源代码提交：`{ctx['build_identity']['commit']}`。Rust binary SHA256：`{ctx['rust_binary']['sha256']}`。输入工作负载SHA256：`{ctx['workload_sha256']}`。两个Rust后端使用同一二进制，所有suite使用相同输入和容量配置；报告提交晚于测量代码提交。", '',
        '```bash', 'source scripts/env.sh', 'cargo build --release --locked --offline --features flash-attn',
        'python3 benchmarks/stress_suite.py --model "$MODEL_DIR" \\',
        '  --backends flash --repetitions 10 --output-dir benchmarks/stress-rerun-flash',
        'python3 benchmarks/stress_suite.py --model "$MODEL_DIR" \\',
        '  --backends native --include-python --repetitions 3 \\',
        '  --python "$REFERENCE_PYTHON" --reference "$REFERENCE_DIR" \\',
        '  --output-dir benchmarks/stress-rerun-reference',
        'python3 benchmarks/summarize_stress.py \\',
        '  --suites benchmarks/stress-rerun-flash benchmarks/stress-rerun-reference', '```', '',
        '`MODEL_DIR`为本地模型目录，`REFERENCE_DIR`为Python nano-vLLM源码目录，`REFERENCE_PYTHON`为安装参考依赖的虚拟环境解释器。首次构建需先安装依赖；已有缓存才使用--offline。基准脚本默认拒绝覆盖已有压测目录。', '',
        '压测数据在 [benchmarks/stress-results](benchmarks/stress-results/)：逐后端summary、原始gzip JSON、日志和GPU采样；CSV提供每轮计时、延迟、执行计数与输出SHA。每份gzip经解压SHA校验，全部已导出的原始字段无损保留；Rust含逐token间隔，Python仅提供ITL统计。报告生成器再次核对gzip和解压内容SHA。历史单轮/小负载对比见 [ATTENTION.md](ATTENTION.md)，基线重建见 [benchmarks/baselines/round1](benchmarks/baselines/round1/README.md)。', '']
    args.output.write_text('\n'.join(lines))
    print(f'Wrote {args.output}; operational_checks_pass={passed}; flash/native={flash/native:.3f}; flash/python={flash/python:.3f}')


if __name__ == '__main__':
    main()

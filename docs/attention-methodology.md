# Attention 实现、验收与复现方法

实际结果见项目根目录 `ATTENTION.md`。以下数值结论对应本轮保存的验证数据。

## 实现范围

- Rust 仍负责 safetensors 权重、调度、分页 KV、持久工作区、采样、CUDA stream 和 Graph 生命周期。推理可执行文件通过 C ABI 调用 CUDA；不调用 Python，也不依赖 PyTorch、ATen、c10 或 libtorch 运行时。
- 默认 native attention 保留第一轮的有序实现，包括 query 寄存器缓存、常见 head dimension 特化、分页寻址优化及 V 预取。第一轮已完成的单次打包元数据上传、缓存 RoPE、分区采样及调度改动继续保留。
- 可选 Flash 使用上游 forward/split-KV CUDA 内核和 CUTLASS/CuTe Tensor Core primitives。Prefill 支持有因果掩码的 ragged query suffix；decode 将同一 KV head 的 GQA queries 通过 stride/view 分组，不额外复制 query，并使用上游 split-count occupancy heuristic。
- Rust 生成 cumulative query lengths 和实际 KV lengths，将它们纳入同一份元数据上传。输出、LSE 与 split-KV 中间工作区在加载时分配；C ABI adapter 不分配、不同步，decode Graph 重放可读取更新后的设备 KV 长度。Decode 的 launch 上界保持固定，以避免随上下文增长改变已捕获的 launch geometry。
- 目前 Flash adapter 仅支持 BF16 Q/K/V/output、head_dim=128、query heads 不超过 256、KV page size 为 256 的正倍数、上下文不超过 4096。支持 MHA/GQA，要求 query heads 可被 KV heads 整除。不支持 dropout、backward、FP16、ALiBi、softcap、sliding window 或其他 head dimensions。原生默认后端的范围不因此扩大或缩小。
- Flash 编译单元单独使用 `--use_fast_math`；原有 `kernels.cu` 不使用该选项。模型权重/激活及 KV 仍为 BF16，GEMM 路径未更换。分块、Tensor Core 运算、中间舍入和归约顺序与 native attention 不同，不能声称两者逐位等价。

独立研究文件 `kernels/decode_candidate.cuh` 还保留了三阶段保序 attention 和两阶段 split-KV 候选，未接入生产 dispatch。保序方案的已测 micro cases 逐字节匹配 native，但收益取决于形状；split-KV 出现数值差异。两者都没有作为本轮端到端 Flash 性能表中的实现。

## 源码来源与许可

FlashAttention 源码来自 [Dao-AILab/flash-attention v2.8.3](https://github.com/Dao-AILab/flash-attention/tree/060c9188beec3a8b62b33a3bfa6d5d2d44975fab)，固定 commit 为 `060c9188beec3a8b62b33a3bfa6d5d2d44975fab`。保留 forward 所需头文件，MIT 许可原文位于 `kernels/flash_native/vendor/flash-attention/LICENSE`。

CUTLASS 来自该 FlashAttention tag 固定的 [NVIDIA/cutlass commit](https://github.com/NVIDIA/cutlass/tree/dc4817921edda44a549197ff3a9dcf5df0636e7b) `dc4817921edda44a549197ff3a9dcf5df0636e7b`，保留 header-only `include/` 树及 BSD-3-Clause 许可 `kernels/flash_native/vendor/cutlass/LICENSE.txt`。

Vendored FlashAttention 的修改限定于三个文件：`flash.h` 移除 ATen include 和未使用的 dropout RNG 字段；`flash_fwd_kernel.h` 移除 ATen Philox include，为编译期关闭的 dropout 分支提供占位 RNG tuple；`flash_fwd_launch_template.h` 将 c10 错误宏替换为独立 CUDA Runtime 错误检查。所选 forward/split-KV 内核的数学及内存访问代码保持上游实现。自有 `flash_native.cu` 初始化参数并复用上游 split-count heuristic；完整说明和文件哈希在 `kernels/flash_native/README.md`、`vendor-sha256.json`。

Python 对照仍为本地 nano-vLLM，参考 commit `3988556e7965b4f2a7d31e0ea7c09b03a4acc306`。基准脚本只在进程中覆盖 KV block 分配数量，以固定预算；模型、attention、采样及调度使用参考实现。既有源码审计说明本地注释/docstring 改动去除 docstring 后 AST 与 HEAD 相同；这不等于工作区 `git status` 干净，也不应这样表述。

## Benchmark 公平口径

比较三组可执行路径：`benchmarks/baselines/round1` 可重建的第一轮源码快照、当前 release 二进制的 `--attention-backend flash`、Python nano-vLLM。比较脚本为 `benchmarks/compare_attention.py`。当前二进制的默认 native 另做回归；不能把“第一轮快照”误标成最早未优化的 Rust 基线。

- 使用同一台 RTX 5070 Ti、WSL2 Ubuntu 24.04 和同一份本地 Qwen3-0.6B BF16 权重。Rust CUDA 工具链及 Python/PyTorch/FlashAttention 版本应随最终结果保存，不能假定今后安装版本不变。
- 三组读取相同已保存的 token JSON、请求顺序、每请求输出长度、temperature=0.6、ignore_eos=true。性能测试不重新分词。Rust 和 Python RNG 实现不同，因此随机生成 token 不要求逐个相同；固定输出长度和 KV 配额不能改变。
- 共同参数为 block_size=256、num_blocks=128、max_sequences=32、max_model_len=4096、max_batch_tokens=1024，KV 容量为 3.5 GiB，decode CUDA Graph 开启。不得以不同 KV 配额、精度、输出长度或少算 token 获得加速。
- GPU 工作串行执行。正确性检查、kernel microbenchmark、Nsight profiling 与正式计时不并行；正式吞吐关闭 profiler。串行同机对照仍可能受温度、频率、桌面活动影响，因此保留逐轮结果和监控日志。
- 小负载和前缀负载完整同工作负载预热 1 次，再测 3 次；每轮先计算 output token/s，再对轮次取中位数。完整 256 请求使用 `warmup-all-batches.json` 短预热 1 次、正式测 1 次。短预热输出长度 2..33 覆盖 decode batch 32..1；单次结果不能代表统计稳定性。
- 模型加载、分词、输出解码、JSON 写出及单独预热排除在吞吐分母之外。吞吐分母包含运行内的调度、prefill、decode、采样、主机设备传输及同步。监控的 `process_wall_seconds` 包含启动/预热，不应替代 `runs[].stats.elapsed_seconds`。
- 冷缓存项在每轮前清理可复用前缀元数据，保留模型权重、编译缓存和已捕获 Graph。热缓存项显式启用 `--warm-prefix-cache`，预热后保留缓存；应同时核对缓存命中计数。冷缓存不表示清除 GPU 硬件缓存或让 CUDA/PyTorch 重新编译。
- `max_sequences=32` 的调度含义并不完全相同：Rust 限制总 active 请求，Python 参考约束每轮调度数。大负载是在同一配置值下比较不同调度器，不是严格相同的总并发或执行序列。

吞吐核算应逐轮确认 `output_tokens == sum(len(output.token_ids))`，且 `output_tokens_per_second == output_tokens / elapsed_seconds`。这个指标包括 prefill，不能命名为“纯 decode 速度”。`prefill_tokens` 可能包含抢占后的重算，应与原始 input tokens 分列；`decode_steps` 为调度步数，不是 decode token 数。比较重算代价应同时读取三组实际统计，不能只凭 Rust 抢占次数归因。

TTFT 从整批请求提交开始计，包含排队；如报告 p50/p95，先在每轮的请求 TTFT 中求分位数，再对轮次取中位数。不能将 Rust `mean_ttft_ms` 与 Python TTFT p50 直接放在同名列中。ITL 为相邻生成 token 间隔，包含其他请求对调度的影响。

整卡显存来自约每 0.5 秒的 `nvidia-smi` 采样，含进程启动/预热和桌面等其他占用；“增量”只能写为相对启动前整卡读数的差值，不能写成进程 allocator 精确峰值。Rust `model_memory` 是静态权重/KV/workspace 字节统计，Python `peak_allocated`/`peak_reserved` 为 PyTorch allocator 统计，三者不能混为同一指标。Flash 新增 scratch 属于工作区；未经分配器/Graph 专门计量，不能把整卡峰值变化全部归于 CUDA Graph 或 scratch。

## 正确性检查与已知限制

以下是本轮已经保存的检查结论，不是最终性能数字：

| 检查 | Flash 结果 | 含义 |
|---|---|---|
| 原有 4 组、每组 16 个 token 的 HF BF16 eager fixture | eager 全部 64 个 token 匹配，首 token 全匹配 | 仅限固定 fixture，不代表任意提示全序列一致 |
| 原有 CUDA Graph 与 eager 比较 | 匹配 | 固定测试下 Graph 开启路径输出一致 |
| 原有 max_batch_tokens=64 与 1024 比较 | 不一致 | 法国相关句子的第 6 个生成 token 分叉，旧严格检查失败 |
| 扩展 20 个 teacher-forced context 的最后位置 logits | 20/20 top1 与 HF eager 匹配，全部有限 | 比较相同上下文，未要求全词表逐位相同 |
| 扩展 20 组、每组 8 个生成 token 的 Graph/eager 比较 | 20/20 一致 | CLI 确认有 decode steps；未导出逐步 Graph counters/logits |
| 扩展 chunk=64/eager 比较 | 18/20 一致 | 失败项为 `legacy-0-first` 和 `legacy-0-sixth` |
| 两组共享前缀的 cold/warm/eager 比较 | 一致，热缓存命中实际发生 | 没有用未触发缓存的测试冒充热缓存覆盖 |

扩展上下文覆盖原 4 组 prompt、各自 teacher-forced 的前 5 个生成 token、page/chunk 边界 255/256/257、511/512/513、1023/1024/1025、长度 3073，以及两组相同 512-token 前缀。所有输入取自固定 token fixture，不依赖重新分词。Logits CLI 做完整 prefill eager，生成模式检查另行覆盖 Graph、chunk 和 prefix；不能用 logits CLI 的 top1 一致声称 Graph 的每步 logits 一致。

旧 `compare_correctness.py` 的判定仍要求：首 token 匹配、全部固定贪心生成匹配、Graph 一致、chunk 一致。Flash 的 `chunk_invariance=false`，脚本退出码仍为 1；扩展 `validate_attention.py report` 的 `strict_gates_pass=false`、`legacy_strict_pass=false` 也原样保留。Native 继续通过旧严格回归。没有把 18/20 改为通过阈值、没有删除失败输入、没有按 prompt 或 temperature 选择内核，也没有以 kernel 误差容限替代完整模型验收。

归约顺序和 BF16 舍入可能使接近的候选 token 排序发生变化；目前只能将此视为需要调查的数值敏感性，不能把“近 tie”作为验收通过的理由，也不能未经逐步隔离就把全部分叉归因于某一个 kernel。扩展报告保存 RMSE、最大绝对误差、cosine、参考 top1 margin 等供诊断；`margin > 2 * observed_max_error` 只保护同一已测上下文的 argmax，不保证后续自由生成稳定。

独立 Flash kernel smoke test 用 BF16 输入与 CPU FP32 因果 attention 对照，覆盖非连续物理 pages、ragged prefill、GQA decode、不同 split 数和改变 KV lengths 后的 Graph 重放；其绝对误差限 0.003 属于算子测试合同。它与原有逐 token 严格生成验收是不同层面的检查，不能互相替代。

## 可复现命令

以下命令在 WSL 项目目录执行。需保留已记录 SHA-256 的 `benchmarks/baselines/round1` 可重建的第一轮源码快照；`compare_attention.py` 不会自动重建历史版本。复测应换新的输出目录，避免覆盖原始结果。各 GPU 阶段顺序执行，测试期间不要同时运行 benchmark 或 profiler。

```bash
cd /root/nano-vllm-rs
source scripts/env.sh
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
P=/root/nano-vllm/.venv/bin/python

# 默认构建不包含 Flash；显式构建后仍默认选择 native。
cargo build --release --locked --offline --features flash-attn
cargo test --locked --offline --features flash-attn
cargo test --locked --offline --features flash-attn --lib gpu::tests -- --ignored --test-threads=1
cargo clippy --locked --offline --features flash-attn --all-targets -- -D warnings

# 独立重建第一轮源码基线。
cargo build --release --locked --offline \
  --manifest-path benchmarks/baselines/round1/Cargo.toml \
  --target-dir benchmarks/baselines/round1/target

# 串行重新测量第一轮快照、Flash、Python，3 次正式测量。
"$P" benchmarks/compare_attention.py \
  --profiles latency-b1 throughput-b8 mixed-b32 prefix-b8-cold prefix-b8-warm \
  --repetitions 3 --output-dir benchmarks/attention-results-repro

# 完整 256 请求：独立短预热，1 次正式测量。
"$P" benchmarks/compare_attention.py \
  --profiles nano-vllm-256 --repetitions 1 \
  --warmup-workload benchmarks/workloads/warmup-all-batches.json \
  --output-dir benchmarks/attention-results-repro
```

原有严格 fixture 的完整复测如下。它对 native 和 Flash 使用同一批固定 fixture；保存失败退出码不等于将失败改为成功。

```bash
"$P" - <<'PY'
import json, shutil, subprocess
from pathlib import Path
root = Path.cwd()
binary = root / 'target/release/nano-vllm-rs'
python = '/root/nano-vllm/.venv/bin/python'
for backend in ['native', 'flash']:
    dest = root / 'benchmarks/attention-check-repro' / backend
    dest.mkdir(parents=True, exist_ok=False)
    for name in ['transformers_reference.json', 'nano-correctness.json']:
        shutil.copy2(root / 'benchmarks/optimization-results' / name, dest / name)
    fixture = json.loads((dest / 'transformers_reference.json').read_text())
    common = ['--model', '/root/huggingface/Qwen3-0.6B', '--attention-backend', backend]
    for row in fixture['results']:
        i = row['request_id']
        tokens = dest / f'tokens-{i}.json'
        tokens.write_text(json.dumps(row['prompt_token_ids']))
        subprocess.run([str(binary), 'logits', *common, '--eager', '--tokens', str(tokens),
                        '--output', str(dest / f'rust-logits-{i}.json')], check=True)
    for mode, flags in [('eager', ['--eager']), ('graph', []),
                        ('chunked', ['--max-batch-tokens', '64'])]:
        subprocess.run([str(binary), 'bench', *common, *flags,
                        '--workload', 'benchmarks/workloads/correctness.json',
                        '--warmups', '0', '--repetitions', '1',
                        '--output', str(dest / f'rust-correctness-{mode}.json')], check=True)
    status = subprocess.run([python, 'benchmarks/compare_correctness.py',
                             '--results', str(dest)]).returncode
    (dest / 'legacy-exit-code.json').write_text(json.dumps({'exit_code': status}))
    print(backend, 'strict comparison exit code:', status, flush=True)
PY
```

扩展验证使用新目录，且显式关联本次 Flash 的旧严格结果，避免误用历史 passing summary：

```bash
V=benchmarks/attention-validation-repro
"$P" benchmarks/validate_attention.py prepare --suite "$V"
"$P" benchmarks/validate_attention.py hf --suite "$V"
"$P" benchmarks/validate_attention.py rust --suite "$V" --label native --attention-backend native
"$P" benchmarks/validate_attention.py rust --suite "$V" --label flash --attention-backend flash
"$P" benchmarks/validate_attention.py report --suite "$V" \
  --candidate flash --baseline native \
  --legacy-summary benchmarks/attention-check-repro/flash/correctness-summary.json
# 当前 Flash 的最后一条命令返回 1：保留此失败，不改阈值。
```

默认后端可直接以 `target/release/nano-vllm-rs bench --attention-backend native ...` 运行；省略 `--attention-backend` 等效于 native。不需要 Flash 时可用 `cargo build --release --locked --offline` 构建。独立 adapter 的 nvcc 编译和 smoke 命令见 `kernels/flash_native/README.md`。

最终报告应附 `benchmarks/attention-results/` 的各引擎逐轮 JSON、进程日志及 `.monitor.json`，两组原有 `correctness-summary.json` 与退出码，`benchmarks/attention-validation/flash-final-validation.json`、压缩 logits/metadata、二进制和源码/工作负载哈希。当前 `BENCHMARK.md` 与 `OPTIMIZATION.md` 是前两阶段的历史报告，不应覆盖其数字并混淆所测版本。

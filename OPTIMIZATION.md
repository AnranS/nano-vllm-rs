# nano-vllm-rs 性能优化实测

生成时间：2026-09-10T02:42:51.481438+00:00。WSL2：`llm-ubuntu-24.04`；项目：`/root/nano-vllm-rs`。

模型为 Qwen3-0.6B BF16，GPU 为 RTX 5070 Ti 16GB。所有吞吐测试关闭 profiler，两版 Rust 与 Python 串行运行。

## 条件与改动

- 相同 token 输入、固定输出长度、temperature=0.6、ignore_eos=true；KV 128×256 块（3.5 GiB）、max_sequences=32、单次 token 预算 1024、上下文上限 4096。
- Rust 前后使用同一概率采样定义和 seed 推导；Python 随机数实现不同。模型加载、分词、输出解码和预热不计入吞吐。
- 小负载完整预热 1 次、测量 3 次取中位数；完整 256 请求使用覆盖 decode batch 32..1 的短预热、测量 1 次。
- Python 的 max_sequences 约束每轮调度数，Rust 约束总 active 数；Python 对照依然是相同配置值下的实现比较，不是严格等价的调度策略比较。
- Attention 使用常见 head_dim 特化、寄存器保存 query、按物理 KV 块查表及成组预取，并保持原有 V 累加顺序和运行时缩放系数；保留通用维度路径，未更换计算精度或 KV 预算。
- 温度采样改为词表分区归约；RoPE 三角函数在加载时缓存；每轮 8 次元数据上传合并为 1 次，设备缓冲区视图保留分配与 Graph 生命周期。
- 调度按可用 KV 容量接纳新请求，减少新 prefill 对已有 decode 请求的驱逐；保持 FIFO 接纳及并发上限。

## 输出吞吐

单位为输出 token/s，分母包含 prefill、decode、调度、数据传输和同步。

| 负载 | 测量次数 | Rust 优化前 | Rust 优化后 | 加速比 | Python nano-vLLM | 优化后/Python |
|---|---:|---:|---:|---:|---:|---:|
| 单请求 128→128 | 3 | 265.06 | 394.56 | 1.49× | 357.34 | 110.4% |
| 8 请求 128→128 | 3 | 1,430.48 | 2,224.64 | 1.56× | 2,440.74 | 91.1% |
| 32 请求混合长度 | 3 | 1,553.57 | 3,054.15 | 1.97× | 4,757.38 | 64.2% |
| 256 请求混合长度 | 1 | 704.55 | 1,738.34 | 2.47× | 3,416.19 | 50.9% |
| 共享前缀冷缓存 | 3 | 406.13 | 836.59 | 2.06× | 1,994.61 | 41.9% |
| 共享前缀热缓存 | 3 | 629.43 | 1,211.19 | 1.92× | 2,181.00 | 55.5% |

## 完整负载调度

| 指标 | Rust 优化前 | Rust 优化后 | Python |
|---|---:|---:|---:|
| 实际 prefill token（含重算） | 299,187 | 187,478 | 211,144 |
| Decode 调度轮数 | 5,014 | 5,019 | 5,023 |
| 抢占次数 | 281 | 97 | 未记录 |

## 正确性

- 独立 Transformers 验证：4 组输入共 64 个贪心输出 token，全序列一致：True；首 token 一致：True。
- 完整 prefill / 分块 prefill 一致：True；eager / CUDA Graph 一致：True。
- 22 个普通测试、6 个真实 GPU 测试及 Clippy 检查通过。覆盖调度、缓存 RoPE、分区采样及缓冲区生命周期，日志保存在 optimization-results/。
- 实验中的并行 V 归约未通过原有生成一致性检查，因此最终采用保持累加顺序的版本；没有放宽验收标准。

## Profiler 证据与局限

- Nsight Systems 仅捕获预热后的测量范围，并启用 CUDA Graph node 跟踪。其耗时会受追踪开销影响，因此不混入上表。
- 优化前的 8 请求分析中，attention 约占 GPU 内核执行时间的 50.0%；32 请求混合长度下约占 82.0%。两版的矩阵乘内核与调用次数基本相同，优化重点据此放在 attention。
- 在 8 请求的本次追踪中，attention 内核累计执行时间由 367.92 ms 降至 136.59 ms，采样由 35.16 ms 降至 1.55 ms（分区与最终归约合计）。这些是 profiler 观测值，不是端到端加速倍数。
- 同一 128 步负载的 cudaMemcpyAsync 调用从 1152 次降至 256 次，cudaStreamSynchronize 从 1281 次降至 385 次。Memcpy API 时长还包含等待 GPU 的时间，不能当作纯数据拷贝耗时。
- 原始 profiler 汇总保存在 optimization-results/profiles/；完整 .nsys-rep 与 SQLite 在 WSL 项目的 .optimization/profiles/。
- 显存日志是每 0.5 秒整卡采样，包含桌面和其他应用变化；不能当作进程分配器的精确峰值。单次完整负载也不能说明统计稳定性。
- 当前仍为单卡 dense Qwen3、4K 上下文版本；没有张量并行、量化或 HTTP 流式服务。

## 复现

```bash
cd /root/nano-vllm-rs
source scripts/env.sh
cargo build --release --locked --offline
python3 benchmarks/compare_optimization.py --profiles latency-b1 throughput-b8 mixed-b32
python3 benchmarks/compare_optimization.py --profiles nano-vllm-256 --repetitions 1 --warmup-workload benchmarks/workloads/warmup-all-batches.json
python3 benchmarks/compare_optimization.py --profiles prefix-b8-cold prefix-b8-warm
python3 benchmarks/summarize_optimization.py
```

before 使用本机保留的 .optimization/baseline/nano-vllm-rs；after 使用当前 release 可执行程序。对照脚本、保存的工作负载与逐轮 JSON 一同交付。

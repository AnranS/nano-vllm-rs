# nano-vllm-rs 实现与本机 Benchmark

生成时间：2026-09-10T01:13:38.353698+00:00。项目：`/root/nano-vllm-rs`，WSL2 `llm-ubuntu-24.04`。

已实现本地 Qwen3 权重加载、BF16 CUDA 推理、真实分页 KV cache、连续批处理、分块 prefill、共享前缀缓存/LRU、容量压力下抢占重算、EOS/温度采样和 decode CUDA Graph。推理进程不调用 Python 或 PyTorch。

## 环境与方法

- GPU：NVIDIA GeForce RTX 5070 Ti, 16303 MiB, 610.88。
- 模型：本地 Qwen3-0.6B，BF16；Rust 使用 CUDA 13.2/cuBLAS，Python 使用 PyTorch 2.12.1+cu130、FlashAttention 2.8.3.post1。
- 相同保存的 token 输入、输出长度、温度 0.6、ignore_eos=true；两版 RNG 实现不同，不要求随机输出逐 token 相同。
- 共同配置：block_size=256，num_blocks=128（KV 3.5 GiB），max_sequences=32，max_model_len=4096，max_batch_tokens=1024，CUDA Graph 开启。
- Python 参考版本：`3988556e7965b4f2a7d31e0ea7c09b03a4acc306`。基准脚本仅在进程中覆盖 KV 分配数量；模型、attention 和调度均使用原实现。
- 基准脚本未修改参考仓库。本地 nanovllm 工作区含中文注释/docstring 改动；去除 docstring 后的 AST 与 HEAD 相同，审计记录见 reference-source-audit.json。
- 两版串行执行；模型加载、tokenizer、输出解码、JSON 写出和预热不计入吞吐时间。运行内包含调度、前向、采样、主机设备同步。
- 小负载与前缀缓存项：完整同负载预热 1 次、正式测量 3 次，报告中位数。256 请求项：独立短预热 1 次、正式测量 1 次，单次结果不代表统计稳定性。短预热的输出长度为 2..33，覆盖 decode batch 32..1。
- 冷缓存项在每轮前清理可复用前缀元数据，保留已加载权重、编译结果与 CUDA Graph。热缓存项保留前一轮缓存。
- 32 的含义并非完全相同：Rust 限制总 active 请求数，原 Python 调度器限制每次调度数量、总 running 可更多。大负载结果比较相同配置下各自调度器，不等于严格相同总并发。

## 输出吞吐

单位为输出 token/s，分母包括 prefill 和 decode；不是仅 decode 的速度。

| 负载 | 测量次数/版 | 输入/输出 token | Rust | Python nano-vLLM | Rust/Python |
|---|---:|---:|---:|---:|---:|
| 单请求 128→128 | 3 | 128/128 | 267.18 | 355.15 | 75.23% |
| 8 请求 128→128 | 3 | 1,024/1,024 | 1,447.20 | 2,405.82 | 60.15% |
| 32 请求混合长度 | 3 | 9,583/6,318 | 1,519.75 | 4,653.64 | 32.66% |
| 原版 256 请求负载 | 1 | 142,827/133,966 | 603.72 | 3,168.54 | 19.05% |
| 共享前缀：冷缓存 | 3 | 4,744/512 | 404.77 | 1,833.60 | 22.08% |
| 共享前缀：热缓存 | 3 | 4,744/512 | 631.05 | 2,045.39 | 30.85% |

完整负载的 Rust 正式运行发生 281 次抢占，执行 299,187 个 prefill token（原始输入 142,827，含重算）及 5,014 个 decode step。这个结果同时体现当前注意力内核与给定 KV 配额下的调度代价。

## 延迟

TTFT 从整批请求提交开始，包含排队；每轮先对请求取分位数，再对测量轮取中位数。ITL 是相邻输出 token 的间隔，包含同引擎其他请求的调度影响。

| 负载 | Rust TTFT p50/p95 (ms) | Python TTFT p50/p95 (ms) | Rust/Python ITL p50 (ms) |
|---|---:|---:|---:|
| 单请求 128→128 | 7.42/7.42 | 20.84/20.84 | 3.72/2.66 |
| 8 请求 128→128 | 40.51/40.51 | 22.87/22.87 | 5.28/3.15 |
| 32 请求混合长度 | 396.01/710.09 | 130.17/236.89 | 13.88/4.58 |
| 原版 256 请求负载 | 70921.61/160653.58 | 17717.15/35189.75 | 31.40/7.28 |
| 共享前缀：冷缓存 | 373.25/591.43 | 50.66/50.66 | 10.56/3.60 |
| 共享前缀：热缓存 | 145.92/145.92 | 23.04/23.04 | 10.58/3.57 |

## 显存

以下统一使用完整负载进程期间的整卡显存采样，含启动/预热，约每 0.5 秒采样一次；差值减去进程启动前整卡占用。它可能包含桌面或其他应用变化，不能视为精确的 allocator 峰值。

| 引擎 | 启动前整卡 (MiB) | 进程期间整卡峰值 (MiB) | 相对增量 (MiB) |
|---|---:|---:|---:|
| rust | 3,144 | 15,882 | 12,738 |
| python | 1,469 | 6,679 | 5,210 |

原始 Rust JSON 另有 weights/KV/workspace 静态字节数，Python JSON 另有 PyTorch peak_allocated/peak_reserved。它们的统计范围不同，未混为同一个显存指标。

## 正确性与实现范围

- 19 个 CPU/模型加载/调度单元测试通过，3 个真实 GPU 算子测试通过，Clippy（警告视为错误）与 release 构建通过。
- GPU 测试包含非连续物理块、ragged GQA 与独立 CPU attention 逐元素对照，及 GEMM、RMSNorm、RoPE、采样和 Graph 回放。
- 4 组输入、共 64 个贪心输出 token：4/4 组与独立 Transformers BF16 eager 参考完全一致。完整 prefill / 64-token 分块 / CUDA Graph 一致性：True。
- 与原 nano-vLLM 的 argmax 诊断参考：3/4 段完全一致；一段开放式续写在后续 token 有不同选择。首 token 全部一致，四组 logits 余弦相似度范围 0.999539–0.999921，最大绝对误差范围 0.203125–0.500000。未宣称跨实现逐位一致。
- 已根据原版实际 TorchInductor 缓存修正 RMSNorm、Q/K Norm、SiLU 的中间 BF16 舍入。HF eager、FlashAttention 与本项目 attention 的计算/舍入路径仍不同；完整差异保存在 correctness-summary.json。
- 当前支持单 GPU、dense Qwen3、最多 4096 上下文；不支持张量并行、量化、RoPE scaling、sliding window、attention bias 或 HTTP 流式服务。
- 当前 attention 是自有 CUDA 实现，尚未实现 FlashAttention 的高效分块/张量核心路径。实测速度未对齐原版，也未对未测试的模型/硬件做性能承诺。
- 本次源码规模：Rust src/ 共 3,987 行（含测试与 CLI），CUDA 共 304 行；不计第三方依赖、生成代码、工具链。

## 复现与原始数据

```bash
cd /root/nano-vllm-rs
source scripts/env.sh
cargo build --release --locked --offline
python3 benchmarks/run_suite.py --profiles latency-b1 throughput-b8 mixed-b32 --repetitions 3 --warmups 1
python3 benchmarks/run_suite.py --profiles nano-vllm-256 --repetitions 1 --warmups 1 --warmup-workload benchmarks/workloads/warmup-all-batches.json
python3 benchmarks/run_suite.py --profiles prefix-b8-cold prefix-b8-warm --repetitions 3 --warmups 1
python3 benchmarks/summarize.py --require-complete
```

固定工作负载在 benchmarks/workloads/；逐轮结果、生成 token、正确性数据及源文件 SHA-256 在 benchmarks/results/；原始进程日志和显存采样在 .bench-logs/。summary.csv 可以直接导入表格软件。

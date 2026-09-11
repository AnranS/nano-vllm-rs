# nano-vllm-rs 第二轮：原生 FlashAttention

本轮为 Rust 引擎加入直接调用 CUDA 的 FlashAttention 2 后端，无 Python/PyTorch/libtorch 运行时依赖。在本机完整 256 请求实测中，吞吐从上轮 **1,734.4** 提升到 **3,877.8 output token/s（2.24×）**；本次 Python 原版为 **3,269.4 token/s**，Rust Flash 高约 **18.6%**。这是特定模型、设备与工作负载的结果。

**Flash 仍是可选后端，默认保持 native。** Flash 完整 prefill 的旧 64 个 greedy 输出匹配 Transformers，Graph 输出一致；chunk64 仍存在已记录的第 6 个 token 分叉，旧严格验收返回 1。性能提升不代表这一限制已经解决。

## 同机重新测量

单位为 output token/s，越大越好。小负载各预热 1 次、测 3 次取中位数；完整 256 请求采用覆盖 batch 形状的短预热、正式测 1 次。

| 负载 | 上轮 Rust | 本轮 Flash Rust | Python 原版 | 相对上轮 | 相对 Python |
| --- | ---: | ---: | ---: | ---: | ---: |
| latency-b1 | 398.2 | 428.1 | 354.7 | 1.08× | 1.21× |
| throughput-b8 | 2,135.8 | 2,745.7 | 2,460.4 | 1.29× | 1.12× |
| mixed-b32 | 3,065.9 | 5,420.2 | 4,562.6 | 1.77× | 1.19× |
| nano-vllm-256 | 1,734.4 | 3,877.8 | 3,269.4 | 2.24× | 1.19× |
| prefix-b8-cold | 839.7 | 1,733.0 | 1,962.7 | 2.06× | 0.88× |
| prefix-b8-warm | 1,241.5 | 2,436.9 | 2,102.4 | 1.96× | 1.16× |


完整工作负载实际处理 142,827 个输入 token、生成 133,966 个输出 token，256 个请求均完成。18 个结果文件的每请求输出长度、请求 ID、总 token 数、时间顺序及进程退出码均校验；原始数据没有用估算补齐。

| 完整负载阶段 | 上轮 Rust | Flash Rust |
| --- | ---: | ---: |
| 总时间 | 77.240 s | 34.547 s |
| prefill 时间 | 11.689 s | 3.277 s |
| decode 时间 | 65.472 s | 31.210 s |
| 实际 prefill token（含重算） | 187,478 | 187,478 |
| 抢占次数 | 97 | 97 |
| decode 步数 | 5,019 | 5,019 |

两版 Rust 的调度和实际计算 token 数一致，本轮收益来自 attention 执行方式升级。Python 总时间为 40.975 s，实际 prefill 211,144 token、decode 5,023 步；不同调度器的重算成本仍不完全相同。

测试设备为 RTX 5070 Ti 16GB，WSL2 `llm-ubuntu-24.04`，模型 `/root/huggingface/Qwen3-0.6B`，BF16。Rust 1.98.1、CUDA Toolkit 13.2、驱动 610.88；Python 对照使用 torch 2.12.1+cu130、transformers 5.15.1、flash_attn 2.8.3.post1。共同参数为 block256、128 个 KV blocks（3.5 GiB）、max_sequences32、context4096、max_batch_tokens1024，Graph 开启，temperature0.6、ignore_eos=true。

GPU 任务串行；计时排除加载、分词及单独预热，包含 prefill、decode、采样、调度和同步。Rust 限制总 active 请求数，Python 的 max_sequences 限制每轮调度数，因此不能宣称调度执行序列完全一致。完整负载单次测量不代表稳定分布；Python mixed-b32 的 3 次结果为 4,697.6 / 4,222.3 / 4,562.6，波动保留在 JSON/CSV。设备监控数据为整卡采样，不能当作进程精确显存峰值。

## Attention 的成本变化

另行运行 Nsight，捕获同一 batch8 工作负载的一个正式轮次（profiling 时间不用于上方吞吐）。Attention kernel 时间合计：上轮 Rust **136.586 ms**，本轮 Flash **48.744 ms**，Python **54.184 ms**。上轮/Python profile 是第一轮保存的相同工作负载记录，并非本轮重新采集。

本轮 Flash attention 约占 kernel 总时间的 **14.1%**，cuBLAS GEMM 约占 **76.1%**。这说明原来的 attention 性能差距已大幅缩小；不能仅从单个算子 microbenchmark 推导端到端加速。

冷前缀负载仍落后 Python 约 12%。当前 Rust prefill 统一使用 paged split-KV，原版在部分 cold prefill 中使用标准 varlen FlashAttention；这一 dispatch 差异与调度成本值得进一步分析。下一轮可考察 contiguous prefill 路径、GEMM 和 graph/workspace 成本。

## 正确性与保留的限制

- 25 个 CPU/模型元数据测试、6 个显式 GPU 测试通过；native、Flash 和 CPU-only 的 Clippy 检查通过，release 构建完成。
- Flash 独立算子 8 组测试覆盖非连续分页、ragged prefill、GQA/MHA/MQA decode、split1/2/8/32、更新长度后的 Graph 重放。相对独立 CPU FP32 attention 的最大绝对误差为 0.0004685，算子测试阈值为 0.003。
- Native 的旧 64-token/HF、Graph、chunk64 验收仍全部通过。Flash 的旧 eager 64-token/HF 和 Graph 通过，chunk64 不通过；未放宽 `compare_correctness.py`。
- 最终 Flash 二进制的扩展 20 个固定上下文全词表 logits 均有限、20/20 top1 与 HF 相同；Graph/eager 为20/20，chunk64/eager 为18/20。共享512-token前缀的冷/热/eager输出一致，真实触发缓存命中。
- 两个扩展分块失败用例均涉及原法国首都续写（原 prompt 和 teacher-forced 第6步）。该第6步参考 top1 margin 为0，存在 BF16 并列；这是数值敏感性的证据，不能作为严格验收通过的理由。扩展报告保留 `strict_gates_pass=false`。
- 额外修复了模型容量的公开可变性：加载后的 options 只能只读访问，防止修改容量后让已捕获 Graph 使用不匹配的页表边界。新 metadata 测试覆盖 cached-prefix 偏移、分组乱序、位置跳跃及上下文边界。

所有性能数字对应显式 `--attention-backend flash`，不能理解为默认 native 已自动达到相同性能。Flash 的实验结果与已知数值限制一并交付，未宣布它通过全部既有严格验收。

## 使用与复现

```bash
cd /root/nano-vllm-rs
source scripts/env.sh
cargo build --release --locked --offline --features flash-attn
./target/release/nano-vllm-rs generate \
  --model /root/huggingface/Qwen3-0.6B \
  --attention-backend flash --chat --prompt '解释 KV cache。' --max-tokens 128
```

`generate`、`bench`、`logits` 共用后端参数。当前 Flash 限 BF16、head_dim128、query heads≤256、block_size为256的倍数、context≤4096，实际在SM120验证。省略后端参数仍使用 native；没有编译 feature 时会明确拒绝 flash。

完整 benchmark/旧严格验收/扩展数值诊断复现命令，以及源码来源与接口说明见 [方法文档](docs/attention-methodology.md) 和 [原生适配器](kernels/flash_native/README.md)。FlashAttention固定v2.8.3 commit `060c9188beec3a8b62b33a3bfa6d5d2d44975fab`（MIT）；CUTLASS固定commit `dc4817921edda44a549197ff3a9dcf5df0636e7b`（BSD-3-Clause）。源码、原许可及变更说明均包含在项目中。

原始性能数据：`benchmarks/attention-results/`；最终数值报告：`benchmarks/attention-validation/flash-final-validation.json`。测试二进制 SHA 与代码/工作负载 SHA 保存在交付清单；旧 `BENCHMARK.md`、`OPTIMIZATION.md` 及原始记录未覆盖。

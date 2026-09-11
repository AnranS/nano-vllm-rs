# nano-vllm-rs

一个可在本机 WSL2 上运行的 Rust + CUDA 单卡 Qwen3 推理引擎，参考 [nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) 的模型执行和调度设计。直接加载本地 Hugging Face safetensors 权重与 tokenizer，使用 cuBLAS 做矩阵乘，使用自有 CUDA 核函数执行 RMSNorm、RoPE、分页 attention 和采样。

**Rust 推理进程不依赖 Python、PyTorch 或 libtorch。** Python 脚本仅用于准备基准工作负载、运行参考实现和比较正确性。原来的模拟调度示例保留在 `demo` 子命令中。

目前验证的模型是 `/root/huggingface/Qwen3-0.6B`。支持 dense Qwen3、单张 NVIDIA GPU、BF16 计算；其他模型大小受显存容量限制，尚未逐一验证。第二轮 FlashAttention 实测与数值限制见 [ATTENTION.md](ATTENTION.md)，原始记录在 `benchmarks/attention-results/`；[OPTIMIZATION.md](OPTIMIZATION.md) 和 [BENCHMARK.md](BENCHMARK.md) 保留前两版结果。

## 连续压测与对比

最新的同进程重复压测、逐轮吞吐与延迟、输出一致性及显存采样见 [STRESS_TEST.md](STRESS_TEST.md)。本轮压测脚本是 `benchmarks/stress_suite.py`，使用保存的固定 token 工作负载，原始 JSON 无损压缩为 gzip 后提交。

```bash
source scripts/env.sh
cargo build --release --locked --offline --features flash-attn
python3 benchmarks/stress_suite.py \
  --model /path/to/Qwen3-0.6B \
  --backends flash --repetitions 10 \
  --output-dir benchmarks/stress-rerun-flash
python3 benchmarks/stress_suite.py \
  --model /path/to/Qwen3-0.6B \
  --backends native --include-python --repetitions 3 \
  --python /path/to/nano-vllm/.venv/bin/python --reference /path/to/nano-vllm \
  --output-dir benchmarks/stress-rerun-reference
```

每个后端在一个模型进程内完成所有重复轮次。脚本核对请求 ID、输出长度、总 token 数、计时值以及同后端跨轮输出 SHA256；不要求 Rust 和 Python 的随机输出相同。GPU 显存为整卡轮询值，不能据此证明分配器没有泄漏。当前验证是 CLI 推理负载，项目尚无 HTTP 服务接口。

第一轮对照所需源码已包含在 [benchmarks/baselines/round1](benchmarks/baselines/round1/README.md)，可以独立构建；复现实测不再依赖本机隐藏的历史二进制。原有 [ATTENTION.md](ATTENTION.md)、[OPTIMIZATION.md](OPTIMIZATION.md)、[BENCHMARK.md](BENCHMARK.md) 保留各轮历史口径。

## 可选 FlashAttention 后端

默认 `native` 保留原有 CUDA attention。可以离线编译并显式启用 FlashAttention 2 原生内核：

```bash
source scripts/env.sh
cargo build --release --locked --offline --features flash-attn
./target/release/nano-vllm-rs generate \
  --model /root/huggingface/Qwen3-0.6B \
  --attention-backend flash --chat --prompt '解释连续批处理。' --max-tokens 128
```

`generate`、`bench`、`logits` 均接受 `--attention-backend native|flash`。编译 `flash-attn` 后仍默认选择 `native`；未编译该 feature 时选择 `flash` 会明确报错。当前 Flash 接口支持 BF16、head_dim=128、query heads≤256、block_size 为 256 的倍数、context≤4096；实际 GPU 验证使用 SM120。原版 FlashAttention/CUTLASS 的必要头文件和许可证已随源码提供，推理运行不加载 Python 或 PyTorch。

**Flash 后端尚未通过严格的分块输出一致性门槛，因此保留为可选路径。** 完整 prefill 的原有 64 个 greedy 输出与 Transformers 一致，Graph 与 eager 一致；chunk64 在法国首都续写的第 6 个 token 分叉。扩展 20 个上下文的最终位置 top-1 均与 Transformers 相同，但其中两个涉及同一续写的用例存在分块差异。没有放宽原验收或用性能提升替代正确性结论，详情与原始 logits 见报告。

## 本机快速开始

项目位于 WSL2 发行版 `llm-ubuntu-24.04` 的 `/root/nano-vllm-rs`。先在 PowerShell 中进入：

```powershell
wsl -d llm-ubuntu-24.04 --cd /root/nano-vllm-rs
```

在 WSL 的 bash 或 zsh 中执行：

```bash
source scripts/env.sh
cargo build --release --locked --offline

./target/release/nano-vllm-rs generate \
  --model /root/huggingface/Qwen3-0.6B \
  --chat \
  --prompt '用三句话解释什么是 KV cache。' \
  --max-tokens 128 \
  --output benchmarks/results/chat-example.json
```

`--chat` 使用 Qwen3 的用户/助手消息格式并关闭 thinking。省略它时，`--prompt` 作为原始文本续写。输出文本写入标准输出，加载信息和统计写入标准错误；`--output` 可选，保存生成 token、文本和计时结果。

可以重复传入 `--prompt`，在一次调用中提交多个请求：

```bash
./target/release/nano-vllm-rs generate \
  --model /root/huggingface/Qwen3-0.6B --chat \
  --prompt '解释 Rust 的所有权。' \
  --prompt '解释连续批处理的作用。' \
  --temperature 0.6 --seed 42 --max-tokens 128
```

`--temperature 0` 是贪心解码，也是默认值；大于零时使用 Gumbel-max 概率采样。采样 seed 按请求 ID 和已生成 token 数推导，不随调度顺序、缓存命中或抢占重算而推进。默认在配置的 EOS token 处停止，`--ignore-eos` 用于要求固定输出长度的基准。当前 CLI 在整个请求批次完成后返回文本，尚无流式输出。

## 构建与检查

本机已配置 Rust **1.98.1**、CUDA Toolkit **13.2** 和 NVIDIA GeForce **RTX 5070 Ti**。工具链版本由 `rust-toolchain.toml` 固定；`source scripts/env.sh` 优先启用项目 `.tools/` 中的 Rust 安装。

首次准备依赖需要联网：

```bash
# 已有兼容工具链时可以跳过 bootstrap。
bash scripts/bootstrap.sh
source scripts/env.sh
cargo fetch --locked
cargo build --release --locked --offline
```

Linux/WSL2 构建需要 C/C++ 链接器、`ar`、CUDA Toolkit、cuBLAS，以及可用的 NVIDIA 驱动。`bootstrap.sh` 安装 Rust 工具链；CUDA 和系统编译工具需要事先准备。`cuda` 是默认 feature，构建脚本默认从 `/usr/local/cuda` 查找 CUDA，默认目标架构为 `sm_120`。其他安装位置或 GPU 架构可以显式指定：

```bash
CUDA_HOME=/usr/local/cuda CUDA_ARCH=120 \
  cargo build --release --locked --offline
```

依赖与工具链已缓存后，可以离线执行常规检查：

```bash
bash scripts/check.sh
```

脚本依次执行格式检查、Clippy、库单元测试和 release 构建。默认的库测试不会执行标记为 `ignored` 的 GPU 测试；显式运行这些测试：

```bash
cargo test --locked --offline --lib gpu::tests \
  -- --ignored --test-threads=1
```

只检查 CPU 调度逻辑和模拟示例时，不需要 CUDA Toolkit 或 GPU：

```bash
cargo test --locked --offline --no-default-features --lib
cargo run --locked --offline --no-default-features -- demo
```

关闭默认 feature 后，`generate`、`bench`、`logits` 会报告未启用 CUDA；当前没有真实模型的 CPU 推理后端。

## 基准测试

基准使用预先保存的 token ID 工作负载，避免把分词、文本解码和模型加载计入吞吐。以下命令预热一次，测量三次，结果保存为 JSON：

```bash
./target/release/nano-vllm-rs bench \
  --model /root/huggingface/Qwen3-0.6B \
  --workload benchmarks/workloads/throughput-b8.json \
  --warmups 1 --repetitions 3 \
  --output benchmarks/results/rust-throughput-b8.json
```

`benchmarks/workloads/` 中的工作负载由 `benchmarks/make_workloads.py` 生成。已有 JSON 文件可直接供 Rust 使用；重新生成时，需要装有 Transformers 的 Python 环境。本机可使用参考项目的虚拟环境：

```bash
/root/nano-vllm/.venv/bin/python benchmarks/make_workloads.py \
  --model /root/huggingface/Qwen3-0.6B \
  --output benchmarks/workloads
```

工作负载覆盖单请求延迟、8 请求吞吐、32 请求混合长度、256 请求队列以及共享前缀。性能工作负载使用固定 seed、`temperature=0.6` 和 `ignore_eos=true`。随机 token 工作负载用于测量计算与调度成本，不用于评价文本质量。

自定义工作负载的格式如下；`prompt_token_ids` 必须来自所选模型的词表，请求 ID 不能重复：

```json
{
  "name": "example",
  "seed": 0,
  "requests": [
    {
      "id": 0,
      "prompt_token_ids": [100, 200, 300],
      "max_tokens": 32,
      "temperature": 0.6,
      "ignore_eos": true
    }
  ]
}
```

默认在每次预热和测量前清空前缀缓存，CUDA Graph 保留。测量热前缀复用时增加 `--warm-prefix-cache`；比较不使用 Graph 的执行时增加 `--eager`。不要混用冷缓存与热缓存结果来计算加速比。

默认预热执行与测量相同的完整工作负载。`--warmup-workload PATH` 可以指定同格式的独立预热 JSON，输出中会记录该路径。大工作负载可使用 `benchmarks/workloads/warmup-all-batches.json` 的短请求覆盖 decode batch 形状；预热内容和测量次数应与结果一同报告。

每次测量记录输出 token/s、输入/输出 token 数、prefill/decode token 和步数、缓存命中、抢占次数及每个请求的 TTFT、总延迟和 token 间隔。TTFT 从整批请求提交时开始，包含排队等待；吞吐分母包含 prefill、decode 和调度开销，排除模型加载及单独的预热调用。设备同步用于计时边界。设置 `--warmups 0` 时，首次执行和 Graph 捕获会进入测量时间。

Python 对照命令使用相同模型、工作负载、KV 容量和调度配置值。两版对 `max_sequences` 的约束不同，详见实测报告：

```bash
/root/nano-vllm/.venv/bin/python benchmarks/python_reference.py \
  --reference /root/nano-vllm \
  --model /root/huggingface/Qwen3-0.6B \
  --workload benchmarks/workloads/throughput-b8.json \
  --warmups 1 --repetitions 3 \
  --output benchmarks/results/python-throughput-b8.json
```

此脚本在当前 Python 进程中覆盖 KV 分配数量以匹配 Rust 容量，不修改参考项目源码。参考实现和 Rust 的概率采样器不同，同名 seed 不意味着二者生成逐 token 相同的序列。完整对照条件、实际结果和局限记录在 [BENCHMARK.md](BENCHMARK.md)，不能从支持相同功能推断性能相同。

也可使用套件脚本串行运行两个引擎，避免它们同时争用 GPU。以下命令对较小工作负载各测三次，并使用各自完整工作负载预热一次：

```bash
/root/nano-vllm/.venv/bin/python benchmarks/run_suite.py \
  --profiles latency-b1 throughput-b8 mixed-b32 prefix-b8-cold prefix-b8-warm \
  --warmups 1 --repetitions 3
```

完整 256 请求工作负载可单独测量一次，使用独立短预热：

```bash
/root/nano-vllm/.venv/bin/python benchmarks/run_suite.py \
  --profiles nano-vllm-256 --warmups 1 --repetitions 1 \
  --warmup-workload benchmarks/workloads/warmup-all-batches.json
```

一次测量不能说明波动范围。套件将进程日志和 GPU 轮询记录写入 `.bench-logs/`，原始推理统计写入 `benchmarks/results/`；轮询显存是整张设备的采样值，包含其他应用和启动开销，不等同于 CUDA 分配器精确峰值。`BENCHMARK.md` 单独汇总实际运行条件与结果。

## 正确性验证

`logits` 导出一个 prompt 最后位置的全词表 logits，供独立实现比较。输入文件是 token ID 数组：

```bash
./target/release/nano-vllm-rs logits \
  --model /root/huggingface/Qwen3-0.6B \
  --tokens benchmarks/workloads/logits-0.json \
  --output benchmarks/results/rust-logits-0.json

/root/nano-vllm/.venv/bin/python benchmarks/transformers_reference.py \
  --model /root/huggingface/Qwen3-0.6B \
  --workload benchmarks/workloads/correctness.json \
  --output benchmarks/results/transformers_reference.json
```

logits 以 BF16 计算，再转换为 FP32 写入 JSON。比较时应同时检查数值误差、argmax 和生成序列，并单独比较完整 prefill、分块 prefill 和 CUDA Graph 路径。

本机已通过 25 个不执行 GPU 运算的库测试，以及 6 个显式 GPU 核函数测试。默认 native 后端的 4 组验证 prompt 各生成 16 个贪心 token，共 64 个 token 与 Hugging Face Transformers 参考输出一致；这组输入下完整 prefill、分块 prefill 和 CUDA Graph 输出也一致。Flash 后端另有 8 组独立 attention 数值测试，其完整模型限制见上文。当前记录在 `benchmarks/attention-results/`，历史记录仍保留。

原 nano-vLLM 的独立对照中，有 1 组开放式续写与 Rust 不同，差异保留在验证结果中。

不同 BF16 内核的归约顺序和舍入点可能不同，尤其当候选 token 的分数接近时，会改变后续开放式续写。上述有限输入的通过结果不构成任意 prompt 的逐位一致性保证，也不保证与原 nano-vLLM 的所有生成序列相同。CPU 模拟 token 测试只验证调度和缓存映射，真实模型正确性依靠单独的 GPU 与参考模型比较。

## 容量与支持范围

`generate`、`bench`、`logits` 共用以下参数：

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `--device` | `0` | CUDA 设备编号 |
| `--attention-backend` | `native` | `flash` 为需要额外编译 feature 的可选原生加速路径 |
| `--block-size` | `256` | 每个 KV 物理块的 token 数 |
| `--num-blocks` | `128` | GPU KV 物理块数 |
| `--max-sequences` | `32` | 同时活动的请求上限 |
| `--max-model-len` | `4096` | prompt 加请求的最大输出长度上限 |
| `--max-batch-tokens` | `1024` | 单次 forward 的 token 预算 |
| `--eager` | 关闭 | 指定后禁用 CUDA Graph |
| `--no-prefix-cache` | 关闭 | 指定后禁用前缀复用 |

`max_sequences` 不能超过 `max_batch_tokens`。超过单次 token 预算的 prompt 会分块 prefill；每个请求仍必须在独占 KV 池时容纳其完整上下文，否则提交时返回容量错误。KV 池在模型加载时分配，物理块在调度过程中按需分配给请求；显存不足时可以先减小 `--num-blocks`，工作区过大时再减小并发或单次 token 预算。

BF16 KV 池的字节数为：

```text
2 × 层数 × num_blocks × block_size × KV 头数 × head_dim × 2
↑ K/V 两份                                               ↑ BF16 字节数
```

例如本地 Qwen3-0.6B 配置与默认参数对应 `2 × 28 × 128 × 256 × 8 × 128 × 2` 字节，即 3.5 GiB。总显存还包括模型权重、临时工作区、CUDA/cuBLAS 和 Graph 资源；输出 JSON 的 `model_memory` 分别记录引擎计算的权重、KV 和工作区字节数，这不是驱动层峰值显存测量。

当前已经实现：

- dense Qwen3 权重与 tokenizer 加载；支持单文件、分片 safetensors，F16/F32 权重加载后转换成 BF16。
- 每层真实分页 KV、因果 GQA、Q/K RMSNorm、RoPE、SiLU MLP 和最终采样。
- 批次内连续接纳与回收、prefill/decode 调度、分块 prefill、混合长度请求。
- 完整块前缀缓存、逐 token 前缀校验、引用计数和未引用块的 LRU 淘汰。
- KV 压力下抢占和重算，保留已生成 token；EOS、温度和确定性采样 seed。
- decode CUDA Graph、固定设备缓冲区、同步计时及 JSON 基准结果。

目前不支持多 GPU/张量并行、量化、MoE、滑动窗口、RoPE scaling、带 attention bias 的模型、HTTP 服务、流式输出、运行中追加或取消请求。`head_dim` 必须为不超过 256 的正偶数；当前 attention 实现的最大上下文为 4096。后续工作见 [开发路线](docs/roadmap.md)。

## 项目结构

```text
src/
  cli.rs                    generate / bench / logits / demo 命令
  model.rs                  Qwen3 权重加载、模型 forward、Graph 管理
  gpu.rs                    CUDA FFI、设备缓冲区与 Graph 生命周期
  runtime.rs                真正的批量推理调度与物理 KV 块管理
  runtime/tests.rs          使用 KV 模拟器验证调度与缓存映射
  engine/, backend/         保留的早期 CPU 模拟示例
kernels/kernels.cu          CUDA 核函数、cuBLAS 和 CUDA Graph C 接口
build.rs                    CUDA 编译及链接配置
benchmarks/                 工作负载生成、Python 对照、验证与原始结果
scripts/                   工具链初始化、环境与检查脚本
docs/roadmap.md             已完成能力和后续工作
BENCHMARK.md               本机实测报告
```

真实推理库入口为 `runtime::{LlmEngine, ModelRunner, Request, SchedulerConfig}` 和 `model::{QwenModel, ModelOptions}`。`LlmEngine::generate` 同步执行一批请求并返回按输入顺序排列的输出；`clear_prefix_cache` 可在测量之间清除缓存。模型执行失败会回收请求持有的块、清空前缀缓存并使当前引擎失效，恢复时应重新创建引擎。

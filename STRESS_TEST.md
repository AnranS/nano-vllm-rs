# 连续压测与 Python 对比报告

本次使用同一 Rust release 二进制，Flash 连续运行 **10 轮**，native运行 **3 轮**，Python运行 **3 轮**；每个后端仅加载模型一次。每轮提交 256 个请求、输入 142,827 token、固定生成 133,966 token。

Flash 吞吐中位数为 **3,966.1 output token/s**，native 为 **1,738.0**，Python 为 **3,340.3**。Flash 为 native 的 **2.28×**，相对 Python **+18.7%**。

运行完整性与跨轮输出校验：**通过**。此结果不替代模型数值验收：Flash 的 chunk64 生成仍有已记录的分叉，默认保持 native，详见 [ATTENTION.md](ATTENTION.md)。

## 条件与计时口径

2026-09-11，本机 WSL2 Ubuntu 24.04、RTX 5070 Ti 16GB、Qwen3-0.6B、BF16。Rust 1.98.1 / CUDA Toolkit 13.2 / 驱动 610.88；Python参考使用 torch 2.12.1+cu130、transformers 5.15.1、flash_attn 2.8.3.post1。

共同参数：block_size=256、num_blocks=128（KV 3.5 GiB）、max_sequences=32、max_model_len=4096、max_batch_tokens=1024；Graph开启、temperature=0.6、ignore_eos=true、固定seed与请求次序。每轮清空可复用前缀元数据，保留模型与Graph。使用 warmup-all-batches.json 单独预热一次。

三个后端串行执行，正式计时期间没有并行GPU测试、Nsight或编译任务。每个后端的重复轮次都在同一模型进程内完成；模型加载、JSON写出与单独预热不计入吞吐，prefill、decode、调度、采样、传输和同步计入。当前是CLI队列负载，未测HTTP/QPS服务。

Rust限制总active请求数，Python限制每轮调度数，因此相同max_sequences并不代表完全相同的调度过程。TTFT从整批提交时开始，包含排队。TTFT/ITL先在每轮内求p50/p95，再对轮次取中位数；Python ITL采用其从真实间隔计算的stats分位数，没有用mean_tpot伪造分位数。

## 吞吐与延迟

| 后端 | 轮数 | 计时合计 s | 吞吐中位数 token/s | 吞吐范围 | CV | TTFT p50 / p95 ms | ITL p50 / p95 ms |
| --- | ---: | ---: | ---: | --- | ---: | --- | --- |
| Rust native | 3 | 231.2 | 1,738.0 | 1,733.8–1,744.2 | 0.24% | 30,240.7 / 61,445.9 | 13.571 / 15.796 |
| Rust Flash | 10 | 338.3 | 3,966.1 | 3,916.3–3,981.8 | 0.48% | 13,673.1 / 27,703.2 | 6.635 / 7.135 |
| Python nano-vLLM | 3 | 119.7 | 3,340.3 | 3,300.2–3,435.8 | 1.69% | 16,104.8 / 32,951.5 | 6.980 / 7.898 |

CV为轮次吞吐的总体标准差/均值。连续短测的波动不等于跨机器或长时间稳定性保证；本次没有构造统计显著性结论。

### 每轮吞吐

| 轮次 | Rust native | Rust Flash | Python |
| ---: | ---: | ---: | ---: |
| 1 | 1,744.2 | 3,974.9 | 3,435.8 |
| 2 | 1,738.0 | 3,981.8 | 3,340.3 |
| 3 | 1,733.8 | 3,977.7 | 3,300.2 |
| 4 | — | 3,946.5 | — |
| 5 | — | 3,956.8 | — |
| 6 | — | 3,916.3 | — |
| 7 | — | 3,970.0 | — |
| 8 | — | 3,940.7 | — |
| 9 | — | 3,964.8 | — |
| 10 | — | 3,967.4 | — |

## 运行完整性与确定性

| 后端 | 完成轮次 / 请求 | 输出总 token | 进程退出码 | 同后端跨轮输出 SHA256 |
| --- | --- | ---: | ---: | --- |
| Rust native | 3 / 768 | 401,898 | 0 | 一致 |
| Rust Flash | 10 / 2,560 | 1,339,660 | 0 | 一致 |
| Python nano-vLLM | 3 / 768 | 401,898 | 0 | 一致 |

校验包含每请求ID唯一且齐全、每请求输出长度、输入/输出总token、有限非负延迟，以及 output_tokens/elapsed_seconds 与记录吞吐相符。Rust额外校验逐请求ITL数量，Python核验其从实际间隔计算的stats分位数有限有效。输出SHA按ID排序后只哈希token，排除时间。Rust和Python的随机采样实现不同，未要求跨后端token相同。

默认native的旧64-token/HF、Graph、chunk64回归通过。Flash的旧eager64-token/HF与Graph通过，chunk64失败；扩展20个固定上下文top1均与HF一致，Graph20/20、chunk18/20。既有严格失败保持原样，没有为了发布修改门槛。

## 显存采样

| 后端 | 整卡峰值 MiB | 末1/4采样中位 MiB | 末1/4范围 MiB | 后1/2线性斜率 MiB/min |
| --- | ---: | ---: | ---: | ---: |
| Rust native | 7,120.0 | 7,102.0 | 5,129.0 | -73.361 |
| Rust Flash | 7,128.0 | 7,110.0 | 18.0 | 0.087 |
| Python nano-vLLM | 7,186.0 | 7,168.0 | 1.0 | -2.626 |

这些是约0.5秒一次的nvidia-smi整卡采样，包含其他应用、模型启动、预热和退出边界，不是每轮分配器精确峰值。斜率只是描述统计；即使近零，也不能据此宣布无内存泄漏。本次覆盖同进程重复推理，不覆盖反复加载/销毁模型或Graph的长期寿命测试。

## 复现与证据

测量时源代码提交：`1539bbc68ce390765ea55dedfd6bc78876e5c3de`。Rust binary SHA256：`7fd36add4c2957370a6ee865b64cc7ee0f304994f920de9f397ab04a91caf336`。输入工作负载SHA256：`45d3e3c5d60ee8feea663b13ca15c3a6f038a4997ce83caf4913abcdb8ab663e`。两个Rust后端使用同一二进制，所有suite使用相同输入和容量配置；报告提交晚于测量代码提交。

```bash
source scripts/env.sh
cargo build --release --locked --offline --features flash-attn
python3 benchmarks/stress_suite.py --model "$MODEL_DIR" \
  --backends flash --repetitions 10 --output-dir benchmarks/stress-rerun-flash
python3 benchmarks/stress_suite.py --model "$MODEL_DIR" \
  --backends native --include-python --repetitions 3 \
  --python "$REFERENCE_PYTHON" --reference "$REFERENCE_DIR" \
  --output-dir benchmarks/stress-rerun-reference
python3 benchmarks/summarize_stress.py \
  --suites benchmarks/stress-rerun-flash benchmarks/stress-rerun-reference
```

`MODEL_DIR`为本地模型目录，`REFERENCE_DIR`为Python nano-vLLM源码目录，`REFERENCE_PYTHON`为安装参考依赖的虚拟环境解释器。首次构建需先安装依赖；已有缓存才使用--offline。基准脚本默认拒绝覆盖已有压测目录。

压测数据在 [benchmarks/stress-results](benchmarks/stress-results/)：逐后端summary、原始gzip JSON、日志和GPU采样；CSV提供每轮计时、延迟、执行计数与输出SHA。每份gzip经解压SHA校验，全部已导出的原始字段无损保留；Rust含逐token间隔，Python仅提供ITL统计。报告生成器再次核对gzip和解压内容SHA。历史单轮/小负载对比见 [ATTENTION.md](ATTENTION.md)，基线重建见 [benchmarks/baselines/round1](benchmarks/baselines/round1/README.md)。

# 开发路线

当前项目已经能在 WSL2 的单张 NVIDIA GPU 上执行真实 dense Qwen3 推理。已实现能力以代码和验证记录为准，最新连续压测见 [STRESS_TEST.md](../STRESS_TEST.md)，Flash 后端对比见 [ATTENTION.md](../ATTENTION.md)，第一轮结果记录在 [OPTIMIZATION.md](../OPTIMIZATION.md)，[BENCHMARK.md](../BENCHMARK.md) 保留初版结果。以下未完成条目是后续工作方向，不代表当前已有支持。

## 已完成：单模型、单卡推理

- [x] Rust 命令行与库 API；真实 `generate`、`bench`、`logits` 路径。
- [x] 读取 Qwen3 配置、Rust tokenizer 和本地 safetensors 权重。
- [x] 支持单文件与分片权重，BF16/F16/F32 输入权重统一为 BF16 计算。
- [x] CUDA 设备与内存所有权、缓冲区边界检查、异步核函数调用及同步主机传输。
- [x] cuBLAS BF16 矩阵乘与 FP32 累加。
- [x] embedding、RMSNorm、融合残差归一化、Q/K 归一化与 RoPE。
- [x] 因果分页 GQA、SiLU MLP、最终 logits 和温度采样。

已经验证本地 Qwen3-0.6B。当前限制为单 GPU、dense Qwen3、`head_dim` 不超过 256、上下文不超过 4096；不接受未实现的量化、滑动窗口、RoPE scaling 或 attention bias 配置。

## 已完成：真实 KV 缓存与调度

- [x] 为每一层分配实际 GPU K/V 张量，物理块表与写入槽位一致。
- [x] 按请求增长动态分配物理块，完成后释放引用。
- [x] 批次内连续接纳请求、混合长度 prefill/decode、单次 token 预算。
- [x] 分块 prefill；中间块只计算和写入 KV，到达末尾才采样。
- [x] 完整块前缀复用，逐 token 校验完整上下文，避免哈希碰撞误命中。
- [x] 前缀块引用计数、未引用块 LRU 淘汰，以及淘汰时清理 CPU 缓存键。
- [x] 最后一个 prompt token 至少重新执行一次，恢复当前请求的 logits。
- [x] 内存压力下抢占、保留输出并重算；避免反复抢占造成无进展。
- [x] 单个 EOS token 停止、固定输出长度、温度、独立请求采样 seed。
- [x] 模型错误后释放引用、清空缓存并阻止继续使用失效引擎。

这里的连续批处理用于同一次 `generate` 提交的请求及其等待队列；目前没有运行中追加请求、取消接口或流式回调。完整前缀键按物理块保存，CPU 元数据受 KV 块数和最大上下文限制，但仍有进一步减少复制与查找成本的空间。

## 已完成：验证和基准基础设施

- [x] GPU 固定工作区与按 decode batch 大小复用的 CUDA Graph。
- [x] Graph 持有捕获时使用的设备缓冲区，处理捕获失败和资源回收。
- [x] KV 模拟器验证块映射、共享引用、LRU 淘汰、抢占和调度独立的 seed。
- [x] GPU 核函数测试，覆盖矩阵乘、归一化、RoPE、分页 GQA、采样和 Graph。
- [x] 最后位置 logits 导出与独立 Transformers/nano-vLLM 参考比较脚本。
- [x] 固定 token 工作负载、可选独立预热、重复测量、冷/热前缀缓存设置和原始 JSON。
- [x] Rust/Python 串行对照套件、进程日志与整卡 GPU 状态轮询。
- [x] 输出吞吐、含排队时间的 TTFT、请求延迟、token 间隔及执行计数。

默认 native 后端已验证 4 组 prompt、每组 16 个贪心 token，共 64 个 token 与 Hugging Face Transformers 输出一致；同组完整 prefill、分块 prefill 和 Graph 输出一致。这是有限样本验证，不能推导所有输入都与任一参考实现逐位相同。特别是不同 BF16 内核对分数接近的候选 token 可能作出不同选择，从而使开放续写分叉。性能是否接近原 nano-vLLM 必须看同条件实测。

原 nano-vLLM 独立对照中已有 1 组开放式续写与 Rust 不同，原始差异被保留，不将其表述为完全一致。

## 下一阶段：扩大正确性和稳定性覆盖

- [x] 增加20组固定上下文，覆盖跨页、chunk边界、3073-token长输入及512-token共享前缀；Flash分块生成仍有2个已记录的差异用例。
- [ ] 对更多 dense Qwen3 检查点完成权重加载、logits 和生成验证。
- [ ] 自动化真实模型的单请求/多请求、冷缓存/热缓存、抢占恢复对照。
- [ ] 长时间运行、重复模型加载与 Graph 销毁的显存稳定性测试。
- [ ] 使用 CUDA 内存检查工具覆盖非法元数据、内核越界与异步错误路径。
- [ ] 明确多 EOS token、取消、超时及部分失败的 API 语义并实现测试。

验收条件：保留原始输入、输出与环境记录；把数值容差、argmax 和序列匹配分开报告，避免将调度模拟测试当作模型正确性证明。

## 下一阶段：性能优化

- [x] 使用 Nsight Systems 在预热后的测量范围分析 CUDA 内核和主机 API。
- [x] 优化分页 attention 的寄存器与按块访存，保持浮点累加顺序。
- [x] 元数据由 8 次上传合并为 1 次，并验证设备 view 和 Graph 生命周期安全。
- [x] 两阶段并行采样与预计算 RoPE，保持原采样定义。
- [x] 容量感知 FIFO 接纳，减少新 prefill 对已有 decode 的抢占。
- [x] 接入可选原生 FlashAttention Tensor Core 路径，完成独立算子和模型诊断；未通过全部分块输出一致性门槛，默认仍为native。
- [ ] 调查Flash分块输出差异，优化cold prefill的标准varlen路径。
- [ ] 优化 prefill/decode 混合负载下的调度公平性与尾延迟。
- [ ] 评估前缀键存储与查找开销，以及 Graph shape 数量对资源的影响。
- [ ] 扩展驱动层峰值显存、长时间吞吐和置信区间记录。

验收条件：每个性能改动先通过数值回归，再以相同模型、精度、KV 容量、请求集合和计时口径对照。性能报告必须保留未达到目标或发生退化的情况。

## 尚未实现：扩展模型与服务

- [ ] 超过 4096 的上下文及相应 attention/工作区设计。
- [ ] 量化加载与计算，例如 INT8、INT4、FP8；每种格式需要独立验证。
- [ ] MoE、其他模型架构、滑动窗口及 RoPE scaling。
- [ ] 多 GPU 张量并行、集合通信与跨设备失败处理。
- [ ] 流式输出、运行中追加和取消请求。
- [ ] HTTP/OpenAI 兼容 API、背压、限流和监控。
- [ ] CPU 模型执行后端。

多卡、服务和新模型能力不能以当前单卡测试代替验收；应分别提供可运行实现、故障测试和对应硬件上的验证记录。

# 第一轮优化的可构建源码基线

这是接入 FlashAttention 之前的第一轮优化源码，取自本机保存的
`/root/nano-vllm-rs/.optimization/round1` 快照，用于重建报告中的 `round1`
对照程序。`src/`、`kernels/`、`Cargo.toml`、`Cargo.lock`、`build.rs`
按快照原样保存；Rust 工具链声明和 `scripts/env.sh`、`scripts/bootstrap.sh`
从当前项目补齐。`source-sha256.json` 记录这些文件的 SHA256。

目录只包含构建需要的源文件和说明，不包含历史二进制、编译缓存、模型权重、
压测输出或新加入的 FlashAttention vendor 代码。这里保留第一轮原生 CUDA
attention、采样和调度实现，不随当前后端修改一起更新。

在 Linux/WSL2、CUDA Toolkit、C++ 编译器和指定 Rust 工具链就绪后，从仓库根目录运行：

```bash
source scripts/env.sh
CUDA_ARCH=120 cargo build --release --locked \
  --manifest-path benchmarks/baselines/round1/Cargo.toml \
  --target-dir benchmarks/baselines/round1/target
```

`CUDA_ARCH=120` 对应报告使用的 RTX 5070 Ti；其他 GPU 按自身 SM 架构设置。
如果机器没有 Rust，可先在仓库根目录运行 `bash scripts/bootstrap.sh`。
本目录也保留同样的脚本，便于单独复制目录后安装工具链。首次构建需能下载
Cargo.lock 中锁定的依赖；依赖已缓存后可追加 `--offline`。

`compare_attention.py` 默认使用上述 target 目录的二进制。可显式指定所有外部路径：

```bash
python3 benchmarks/compare_attention.py \
  --model /path/to/Qwen3-0.6B \
  --python /path/to/nano-vllm/.venv/bin/python \
  --reference /path/to/nano-vllm \
  --binary target/release/nano-vllm-rs \
  --baseline-binary benchmarks/baselines/round1/target/release/nano-vllm-rs \
  --profiles latency-b1 throughput-b8 mixed-b32 --repetitions 3
```

该脚本始终串行运行三个引擎，使用相同保存的输入、输出长度和 KV 配额。
`--python` 保留虚拟环境入口，不会解析符号链接后绕过虚拟环境。
若基线尚未构建，脚本会在启动任何引擎前给出完整构建命令。

本次整理仅复制和校验源码；发布前的完整构建与复现实测由主流程验证，
不能仅凭本目录存在就视为已在新机器上完成验证。

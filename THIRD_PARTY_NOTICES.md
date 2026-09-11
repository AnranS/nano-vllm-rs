# Third-party source notices

The following third-party CUDA headers are included so the optional FlashAttention
backend can be built without Python or libtorch. Their original licenses are retained.

| Component | Source revision | License file |
| --- | --- | --- |
| FlashAttention 2 | `060c9188beec3a8b62b33a3bfa6d5d2d44975fab` (`v2.8.3`) | [MIT](kernels/flash_native/vendor/flash-attention/LICENSE) |
| CUTLASS / CuTe | `dc4817921edda44a549197ff3a9dcf5df0636e7b` | [BSD-3-Clause](kernels/flash_native/vendor/cutlass/LICENSE.txt) |

The adapter and the three modified upstream headers are documented in
[the backend source notes](kernels/flash_native/README.md). The vendored-file SHA256
inventory is `kernels/flash_native/vendor-sha256.json`.

The project follows the model execution and scheduling design of
[nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm). Its locally installed
Python implementation is used as an independent benchmark reference and is not
bundled as a runtime dependency. Rust dependencies are pinned in `Cargo.lock`;
their source and license metadata are supplied by their respective crates.

These notices describe the third-party components; they do not declare a project-wide
license for the project's own source.

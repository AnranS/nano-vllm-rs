# Standalone FlashAttention forward backend

This directory contains an inference-only C ABI adapter and vendored CUDA headers.
It builds without Python, PyTorch, ATen, c10, or libtorch. The CUDA kernels use
CUTLASS/CuTe tensor-core primitives; Rust owns the buffers and CUDA stream.

## Scope

- BF16 query, key, value, and output; head dimension exactly 128.
- Paged KV layout `[physical_pages, page_size, kv_heads, 128]`.
- Page size must be a positive multiple of 256; current adapter context limit is 4096.
- Causal ragged/chunked prefill: queries are the contiguous suffix of each KV context.
- Decode: one query per sequence, MHA or GQA with `q_heads % kv_heads == 0`.
- Decode uses a view/stride change to group queries by KV head without copying.
- Split-KV decode uses the upstream occupancy heuristic and caller-owned FP32 scratch.
- No dropout, backward, ALiBi, softcap, sliding window, FP16, or other head dimensions.
- The adapter does not allocate, synchronize, append KV, apply RoPE, or own a CUDA context.

See `flash_native.h` for the complete buffer contract. In particular LSE scratch is
**padded** (`batch * q_heads * max_q` floats), including for ragged prefill. The
caller validates buffer sizes, device identities, nonaliasing, and device metadata.
`nvr_flash_fwd` validates scalar host arguments, and converts CUDA launch errors
to an error code plus a thread-local error string. Asynchronous failures must be
checked by the caller when synchronizing its stream.

## Provenance and licenses

- FlashAttention: https://github.com/Dao-AILab/flash-attention
  - Tag: `v2.8.3`
  - Commit: `060c9188beec3a8b62b33a3bfa6d5d2d44975fab`
  - Included files: `csrc/flash_attn/src` forward `.h` and `.cuh` headers.
  - License: MIT, retained in `vendor/flash-attention/LICENSE`.
- CUTLASS: https://github.com/NVIDIA/cutlass
  - Commit pinned by that FlashAttention tag:
    `dc4817921edda44a549197ff3a9dcf5df0636e7b`
  - Included files: header-only `include/` tree.
  - License: BSD-3-Clause, retained in `vendor/cutlass/LICENSE.txt`.

Only three vendored FlashAttention files are adapted:

1. `flash.h`: remove the ATen header and unused dropout RNG state field.
2. `flash_fwd_kernel.h`: remove the ATen Philox unpack include; use a constant
   dummy RNG tuple in the compile-time-disabled dropout path.
3. `flash_fwd_launch_template.h`: replace c10 CUDA error macros with
   `native_cuda_check.h`, which uses CUDA Runtime errors and standard C++ exceptions.

The math and memory access code of the selected forward/split-KV kernels is
unchanged. `flash_native.cu` initializes the standalone parameter structure and
copies the upstream split-count heuristic. Inference builds must define all five
`FLASHATTENTION_DISABLE_*` flags below; dropout is deliberately not supported.

## Standalone build and numerical checks

Run from this directory (replace architecture for the target GPU as appropriate):

```bash
nvcc -std=c++17 -O3 --use_fast_math -arch=sm_120 \
  --expt-relaxed-constexpr --expt-extended-lambda -Xcompiler=-fPIC \
  -DFLASHATTENTION_DISABLE_DROPOUT -DFLASHATTENTION_DISABLE_ALIBI \
  -DFLASHATTENTION_DISABLE_UNEVEN_K -DFLASHATTENTION_DISABLE_SOFTCAP \
  -DFLASHATTENTION_DISABLE_LOCAL \
  -I. -Ivendor/flash-attention -Ivendor/cutlass/include \
  -c flash_native.cu -o /tmp/nvr_flash_native.o
ar rcs /tmp/libnvr_flash.a /tmp/nvr_flash_native.o
nvcc -std=c++17 -O2 -arch=sm_120 -I. smoke.cu /tmp/libnvr_flash.a \
  -o /tmp/nvr_flash_smoke
/tmp/nvr_flash_smoke
```

The smoke test independently computes CPU FP32 causal attention for BF16 inputs,
including noncontiguous physical pages, ragged prefill, GQA decode with 1/2/8/32
splits, and CUDA Graph replay after changing device KV lengths. It reports maximum
and mean absolute errors and fails for nonfinite output or absolute error > 0.003.
These are kernel tests, separate from complete-model greedy token comparisons.

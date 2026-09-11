#pragma once
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

// Standalone CUDA backend. No Python, ATen, c10, or libtorch dependency.
// All pointers except stream are CUDA device pointers. Tensor strides below
// are contiguous: q/out[total_q,q_heads,128], k/v[pages,block_size,kv_heads,128].
// cu_q[batch+1] stores cumulative query counts; kv_lens[batch] includes this
// step's keys; block_table[batch,table_stride] maps logical to physical pages.
// Causal queries must be the contiguous suffix of each sequence's KV context.
// block_size must be divisible by 256; max_q/max_k bound actual device lengths.
// max_q==1 requires exactly one query for every sequence and activates GQA swap.
// lse needs batch*q_heads*max_q floats (padded even for ragged batches).
// num_splits must be 1 for prefill (max_q>1). Decode may use 1..128 splits.
// For decode and num_splits>1, lse_accum needs num_splits*batch*q_heads floats,
// out_accum needs that count*128 floats. Scratch/output cannot alias any input.
// No allocations or synchronization are performed; graph capture is supported.
// Returns 0 on launch success, -1 on invalid arguments or CUDA host/launch error.
// Asynchronous execution errors remain the caller's responsibility to surface.
int nvr_flash_fwd(void* stream, const void* q, const void* k, const void* v,
                  void* out, const int* cu_q, const int* kv_lens,
                  const int* block_table, int batch, int total_q,
                  int max_q, int max_k, int q_heads, int kv_heads,
                  int block_size, int table_stride, int num_splits,
                  float* lse, float* lse_accum, float* out_accum);

// Pure host heuristic matching FlashAttention v2.8.3 decode dispatch.
// Pass the physical SM count (not multiplied by two). Invalid args return 0.
int nvr_flash_num_splits(int batch, int q_heads, int kv_heads,
                        int max_k, int num_sms);
const char* nvr_flash_last_error(void);

#ifdef __cplusplus
}
#endif

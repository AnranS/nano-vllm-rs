// C ABI integration for the vendored FlashAttention v2.8.3 forward kernels.
// The original kernels retain their MIT license; this adapter owns no tensors.
#include "flash_native.h"
#include "flash_fwd_launch_template.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>

namespace {
thread_local char last_error[512] = {};
int ceildiv(int x, int y) { return (x + y - 1) / y; }
}

extern "C" const char* nvr_flash_last_error() { return last_error; }

extern "C" int nvr_flash_num_splits(int batch, int q_heads, int kv_heads,
                                    int max_k, int num_sms) {
    if (batch <= 0 || q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads ||
        max_k <= 0 || max_k > 4096 || num_sms <= 0 || num_sms > 1024) return 0;
    const int groups = q_heads / kv_heads;
    const int work = batch * kv_heads * ceildiv(groups, 64);
    const int sms = num_sms * 2;
    if (work >= .8f * sms) return 1;
    const int nblocks = ceildiv(max_k, 128);
    const int max_splits = std::min({128, sms, nblocks});
    float best = 0, efficiency[128] = {};
    for (int split = 1; split <= max_splits; ++split) {
        if (split != 1 && ceildiv(nblocks, split) == ceildiv(nblocks, split - 1)) continue;
        const float waves = float(work * split) / sms;
        efficiency[split - 1] = waves / std::ceil(waves);
        best = std::max(best, efficiency[split - 1]);
    }
    for (int split = 1; split <= max_splits; ++split) {
        if (efficiency[split - 1] >= .85f * best) return split;
    }
    return 1;
}

extern "C" int nvr_flash_fwd(void* stream, const void* q, const void* k,
                              const void* v, void* out, const int* cu_q,
                              const int* kv_lens, const int* block_table,
                              int batch, int total_q, int max_q, int max_k,
                              int q_heads, int kv_heads, int block_size,
                              int table_stride, int num_splits, float* lse,
                              float* lse_accum, float* out_accum) {
    last_error[0] = 0;
    if (!q || !k || !v || !out || !cu_q || !kv_lens || !block_table || !lse ||
        batch <= 0 || batch > 4096 || total_q < batch || max_q < 1 || max_q > 4096 ||
        max_k < 1 || max_k > 4096 || q_heads <= 0 || q_heads > 256 ||
        kv_heads <= 0 || q_heads % kv_heads || block_size <= 0 || block_size % 256 ||
        table_stride <= 0 || max_k > int64_t(block_size) * table_stride ||
        total_q > int64_t(batch) * max_q || num_splits < 1 || num_splits > 128 ||
        (max_q == 1 && total_q != batch) || (max_q != 1 && num_splits != 1) ||
        (num_splits > 1 && (!lse_accum || !out_accum))) {
        std::snprintf(last_error, sizeof(last_error), "invalid FlashAttention BF16 hdim128 paged arguments");
        return -1;
    }
    try {
        FLASH_NAMESPACE::Flash_fwd_params p{};
        const bool decode = max_q == 1;
        const bool swap = decode && q_heads > kv_heads;
        const int groups = q_heads / kv_heads;
        p.q_ptr = const_cast<void*>(q);
        p.k_ptr = const_cast<void*>(k);
        p.v_ptr = const_cast<void*>(v);
        p.o_ptr = out;
        p.b = batch;
        p.h = swap ? kv_heads : q_heads;
        p.h_k = kv_heads;
        p.h_h_k_ratio = p.h / p.h_k;
        p.d = p.d_rounded = 128;
        p.seqlen_q = swap ? groups : max_q;
        p.seqlen_k = max_k;
        p.total_q = total_q;
        p.seqlen_q_rounded = ceildiv(p.seqlen_q, 128) * 128;
        p.seqlen_k_rounded = ceildiv(max_k, 128) * 128;
        p.q_batch_stride = p.o_batch_stride = int64_t(q_heads) * 128;
        p.q_row_stride = p.o_row_stride = swap ? 128 : int64_t(q_heads) * 128;
        p.q_head_stride = p.o_head_stride = swap ? int64_t(groups) * 128 : 128;
        p.k_row_stride = p.v_row_stride = int64_t(kv_heads) * 128;
        p.k_head_stride = p.v_head_stride = 128;
        p.k_batch_stride = p.v_batch_stride = int64_t(block_size) * kv_heads * 128;
        p.cu_seqlens_q = decode ? nullptr : const_cast<int*>(cu_q);
        // Noncumulative lengths are identical to flash_attn_with_kvcache.
        p.cu_seqlens_k = const_cast<int*>(kv_lens);
        p.is_seqlens_k_cumulative = false;
        p.seqused_k = const_cast<int*>(kv_lens);
        p.block_table = const_cast<int*>(block_table);
        p.block_table_batch_stride = table_stride;
        p.page_block_size = block_size;
        p.softmax_lse_ptr = lse;
        p.softmax_lseaccum_ptr = lse_accum;
        p.oaccum_ptr = out_accum;
        p.num_splits = num_splits;
        p.scale_softmax = 1.f / std::sqrt(128.f);
        p.scale_softmax_log2 = p.scale_softmax * float(M_LOG2E);
        p.p_dropout = p.rp_dropout = 1.f;
        p.p_dropout_in_uint8_t = 255;
        p.scale_softmax_rp_dropout = p.scale_softmax;
        p.is_bf16 = true;
        p.is_causal = !decode;
        p.window_size_left = decode ? -1 : max_k;
        p.window_size_right = decode ? -1 : 0;
        // Splitkv writes padded LSE; it is scratch only and not exposed to Rust.
        p.unpadded_lse = false;
        p.seqlenq_ngroups_swapped = false;
        if (decode) {
            FLASH_NAMESPACE::run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 128, false>(p, static_cast<cudaStream_t>(stream));
        } else {
            FLASH_NAMESPACE::run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 128, true>(p, static_cast<cudaStream_t>(stream));
        }
        return 0;
    } catch (const std::exception& err) {
        std::snprintf(last_error, sizeof(last_error), "%s", err.what());
        return -1;
    } catch (...) {
        std::snprintf(last_error, sizeof(last_error), "unknown native FlashAttention exception");
        return -1;
    }
}

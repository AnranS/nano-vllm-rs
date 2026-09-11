#pragma once
// Experimental decode kernels. Include after kernels.cu in an independent
// microbenchmark; production dispatch and Rust ABI are deliberately unchanged.
// No allocations occur in these functions, so explicitly owned scratch buffers
// can remain at stable addresses throughout CUDA Graph capture/replay.

template<int DIM>
__device__ __forceinline__ bool decode_metadata_valid(int row, const int* positions,
    const int* sequence_ids, int batch, int stride, int block_size, int max_context) {
    const int sequence = sequence_ids[row];
    const int64_t context = int64_t(positions[row]) + 1;
    return sequence >= 0 && sequence < batch && context > 0 && context <= max_context
        && context <= int64_t(stride) * block_size;
}

// Exact candidate stage 1: distribute independent Q.K scores over multiple
// CTAs. Each score keeps the baseline's dot-product and warp-reduction order.
template<int DIM, bool BLOCK256>
__global__ void decode_scores_kernel(const bf16* q, const bf16* keys,
    const int* positions, const int* seqs, const int* tables, float* scratch,
    int qheads, int kvheads, int block_size, int blocks, int stride, int batch,
    int max_context, int runtime_dim, int partitions) {
    const int row = blockIdx.x, head = blockIdx.y, part = blockIdx.z;
    if (!decode_metadata_valid<DIM>(row,positions,seqs,batch,stride,block_size,max_context)) return;
    const int context = positions[row] + 1, kvhead = head / (qheads / kvheads);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int64_t index = int64_t(row) * qheads + head;
    const int64_t qbase = index * DIM;
    float* scores = scratch + index * (max_context + 1);
    const int* table = tables + int64_t(seqs[row]) * stride;
    float query[DIM / 32];
    #pragma unroll
    for (int d = 0; d < DIM / 32; ++d) query[d] = __bfloat162float(q[qbase + lane + 32*d]);
    const float scale = rsqrtf(float(runtime_dim));
    for (int token = part*8 + warp; token < context; token += partitions*8) {
        const int64_t base = kv_token_base<BLOCK256>(token,table,block_size,blocks,kvheads,kvhead,DIM);
        float dot = 0;
        if (base >= 0) {
            #pragma unroll
            for (int d = 0; d < DIM / 32; ++d) dot += query[d] * __bfloat162float(keys[base + lane + 32*d]);
            dot = warp_sum(dot) * scale;
        } else dot = -INFINITY;
        if (lane == 0) scores[token] = dot;
    }
}

// Exact stage 2: the sum still uses 256 threads with precisely the baseline's
// token-to-thread assignment and block_sum reduction. Keep unnormalised exp
// weights plus the denominator, as normalising before V would change rounding.
template<int DIM>
__global__ void decode_softmax_kernel(const int* positions, const int* seqs, float* scratch,
    int qheads, int block_size, int stride, int batch, int max_context) {
    const int row = blockIdx.x, head = blockIdx.y;
    if (!decode_metadata_valid<DIM>(row,positions,seqs,batch,stride,block_size,max_context)) return;
    const int context = positions[row] + 1;
    float* scores = scratch + (int64_t(row)*qheads + head) * (max_context + 1);
    extern __shared__ float values[];
    float maximum = -INFINITY;
    for (int token = threadIdx.x; token < context; token += blockDim.x) {
        values[token] = scores[token]; maximum = fmaxf(maximum,values[token]);
    }
    maximum = block_max(maximum);
    float sum = 0;
    for (int token = threadIdx.x; token < context; token += blockDim.x) {
        float probability = isfinite(maximum) ? expf(values[token] - maximum) : 0.0f;
        scores[token] = probability; sum += probability;
    }
    const float denominator = block_sum(sum);
    if (threadIdx.x == 0) scores[max_context] = denominator;
}

// Exact stage 3: distribute dimensions rather than reducing partial V sums.
// Every dimension still accumulates token 0,1,2,... with the same FP32 FMAs.
template<int DIM, bool BLOCK256, int TILE_DIM>
__global__ void decode_values_kernel(const bf16* values, const int* positions,
    const int* seqs, const int* tables, const float* scratch, bf16* out,
    int qheads, int kvheads, int block_size, int blocks, int stride, int batch, int max_context) {
    const int row = blockIdx.x, head = blockIdx.y, dimension = int(blockIdx.z)*TILE_DIM + threadIdx.x;
    if (dimension >= DIM) return;
    const int64_t index = int64_t(row)*qheads + head;
    if (!decode_metadata_valid<DIM>(row,positions,seqs,batch,stride,block_size,max_context)) {
        out[index*DIM + dimension] = __float2bfloat16(0.0f); return;
    }
    const int context = positions[row] + 1, kvhead = head/(qheads/kvheads);
    const int* table = tables + int64_t(seqs[row])*stride;
    const float* scores = scratch + index*(max_context + 1);
    const int size = BLOCK256 ? 256 : block_size;
    float value = 0;
    for (int start = 0, logical = 0; start < context; start += size, ++logical) {
        const int block = table[logical], count = min(size,context-start);
        if (block < 0 || block >= blocks) continue;
        const bf16* base = values + (int64_t(block)*block_size*kvheads + kvhead)*DIM + dimension;
        for (int offset = 0; offset < count; offset += 8) {
            float weights[8], input[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) if (offset+j < count) {
                weights[j] = scores[start+offset+j];
                input[j] = __bfloat162float(base[int64_t(offset+j)*kvheads*DIM]);
            }
            #pragma unroll
            for (int j = 0; j < 8; ++j) if (offset+j < count) value += weights[j]*input[j];
        }
    }
    const float denominator = scores[max_context];
    out[index*DIM + dimension] = __float2bfloat16_rn(denominator > 0 ? value/denominator : 0.0f);
}

// Conventional split-KV candidate: one CTA per query/head/context partition,
// then merge partial softmax statistics and V sums. This changes FP32 reduction
// order and must be evaluated separately from the exact staged candidate.
template<int DIM, bool BLOCK256>
__global__ void decode_partition_kernel(const bf16* q, const bf16* keys, const bf16* values,
    const int* positions, const int* seqs, const int* tables, float* scratch,
    int qheads, int kvheads, int block_size, int blocks, int stride, int batch,
    int max_context, int runtime_dim, int partitions) {
    const int row = blockIdx.x, head = blockIdx.y, part = blockIdx.z;
    const int64_t index = int64_t(row)*qheads + head;
    float* partial = scratch + (index*partitions + part)*(DIM + 2);
    const bool valid = decode_metadata_valid<DIM>(row,positions,seqs,batch,stride,block_size,max_context);
    const int context = valid ? positions[row]+1 : 0;
    const int segment = (context + partitions-1)/partitions;
    const int begin = part*segment, end = min(context,begin+segment), count = end-begin;
    if (!valid || count <= 0) {
        if (threadIdx.x < DIM) partial[threadIdx.x] = 0;
        if (threadIdx.x == 0) { partial[DIM] = -INFINITY; partial[DIM+1] = 0; }
        return;
    }
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, kvhead = head/(qheads/kvheads);
    const int* table = tables + int64_t(seqs[row])*stride;
    extern __shared__ float scores[];
    float query[DIM/32];
    #pragma unroll
    for (int d = 0; d < DIM/32; ++d) query[d] = __bfloat162float(q[index*DIM + lane + 32*d]);
    const float scale = rsqrtf(float(runtime_dim));
    float maximum = -INFINITY;
    for (int local = warp; local < count; local += 8) {
        const int64_t base = kv_token_base<BLOCK256>(begin+local,table,block_size,blocks,kvheads,kvhead,DIM);
        float dot = 0;
        if (base >= 0) {
            #pragma unroll
            for (int d = 0; d < DIM/32; ++d) dot += query[d]*__bfloat162float(keys[base + lane + 32*d]);
            dot = warp_sum(dot)*scale;
        } else dot = -INFINITY;
        if (lane == 0) { scores[local] = dot; maximum = fmaxf(maximum,dot); }
    }
    maximum = block_max(maximum);
    float sum = 0;
    for (int local = threadIdx.x; local < count; local += blockDim.x) {
        const float probability = isfinite(maximum) ? expf(scores[local]-maximum) : 0.0f;
        scores[local] = probability; sum += probability;
    }
    const float denominator = block_sum(sum);
    if (threadIdx.x < DIM) {
        const int size = BLOCK256 ? 256 : block_size;
        float value = 0;
        for (int start = begin; start < end;) {
            const int logical = BLOCK256 ? start>>8 : start/block_size;
            const int offset = BLOCK256 ? start&255 : start%block_size;
            const int block = table[logical], length = min(size-offset,end-start);
            if (block >= 0 && block < blocks) {
                const bf16* base = values + ((int64_t(block)*block_size + offset)*kvheads + kvhead)*DIM + threadIdx.x;
                for (int t = 0; t < length; t += 8) {
                    float weights[8], input[8];
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) if (t+j < length) {
                        weights[j] = scores[start-begin+t+j];
                        input[j] = __bfloat162float(base[int64_t(t+j)*kvheads*DIM]);
                    }
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) if (t+j < length) value += weights[j]*input[j];
                }
            }
            start += length;
        }
        partial[threadIdx.x] = value;
    }
    if (threadIdx.x == 0) { partial[DIM] = maximum; partial[DIM+1] = denominator; }
}

template<int DIM>
__global__ void decode_combine_kernel(const float* scratch, bf16* out, int qheads, int partitions) {
    const int64_t index = int64_t(blockIdx.x)*qheads + blockIdx.y;
    const float* partial = scratch + index*partitions*(DIM+2);
    __shared__ float weights[32], denominator;
    if (threadIdx.x == 0) {
        float maximum = -INFINITY;
        for (int p = 0; p < partitions; ++p) maximum = fmaxf(maximum,partial[p*(DIM+2)+DIM]);
        float total = 0;
        for (int p = 0; p < partitions; ++p) {
            const float value = isfinite(maximum) ? expf(partial[p*(DIM+2)+DIM]-maximum) : 0.0f;
            weights[p] = value;
            total += value*partial[p*(DIM+2)+DIM+1];
        }
        denominator = total;
    }
    __syncthreads();
    if (threadIdx.x < DIM) {
        float value = 0;
        for (int p = 0; p < partitions; ++p) value += weights[p]*partial[p*(DIM+2)+threadIdx.x];
        out[index*DIM + threadIdx.x] = __float2bfloat16_rn(denominator > 0 ? value/denominator : 0.0f);
    }
}

template<int DIM, bool BLOCK256>
static int launch_decode_staged(Context* c, const bf16* q, const bf16* k, const bf16* v,
    const int* positions, const int* seqs, const int* tables, bf16* out,
    int rows, int qheads, int kvheads, int block_size, int blocks, int stride,
    int batch, int max_context, float* scratch, int partitions) {
    decode_scores_kernel<DIM,BLOCK256><<<dim3(rows,qheads,partitions),256,0,c->stream>>>(q,k,positions,seqs,tables,scratch,qheads,kvheads,block_size,blocks,stride,batch,max_context,DIM,partitions);
    CUDA(cudaGetLastError());
    decode_softmax_kernel<DIM><<<dim3(rows,qheads),256,size_t(max_context)*4,c->stream>>>(positions,seqs,scratch,qheads,block_size,stride,batch,max_context);
    CUDA(cudaGetLastError());
    decode_values_kernel<DIM,BLOCK256,32><<<dim3(rows,qheads,DIM/32),32,0,c->stream>>>(v,positions,seqs,tables,scratch,out,qheads,kvheads,block_size,blocks,stride,batch,max_context);
    LAUNCH_CHECK;
}
template<int DIM, bool BLOCK256>
static int launch_decode_splitkv(Context* c, const bf16* q, const bf16* k, const bf16* v,
    const int* positions, const int* seqs, const int* tables, bf16* out,
    int rows, int qheads, int kvheads, int block_size, int blocks, int stride,
    int batch, int max_context, float* scratch, int partitions) {
    const size_t score_bytes = size_t((max_context+partitions-1)/partitions)*sizeof(float);
    decode_partition_kernel<DIM,BLOCK256><<<dim3(rows,qheads,partitions),256,score_bytes,c->stream>>>(q,k,v,positions,seqs,tables,scratch,qheads,kvheads,block_size,blocks,stride,batch,max_context,DIM,partitions);
    CUDA(cudaGetLastError());
    decode_combine_kernel<DIM><<<dim3(rows,qheads),256,0,c->stream>>>(scratch,out,qheads,partitions);
    LAUNCH_CHECK;
}

// Both interfaces append (scratch, partitions) to nv_attention's arguments.
// staged scratch: rows*qheads*(max_context+1) floats.
// splitkv scratch: rows*qheads*partitions*(head_dim+2) floats.
// The owner must validate buffer extents/contexts and retain scratch with Graph.
// These experimental entry points intentionally reject unsupported dimensions.
#define DECODE_ARGS Context* c, const bf16* q, const bf16* k, const bf16* v, const int* p, const int* s, const int* t, bf16* o, int rows, int qh, int kvh, int dim, int bs, int blocks, int stride, int batch, int max_context, float* scratch, int partitions
#define DECODE_CALL c,q,k,v,p,s,t,o,rows,qh,kvh,bs,blocks,stride,batch,max_context,scratch,partitions
#define DECODE_DISPATCH(launch) \
    if (partitions < 1 || partitions > 32 || max_context < 1 || max_context > 4096 || rows < 1 || qh < 1 || qh > 65535 || kvh < 1 || qh % kvh || bs < 1 || blocks < 1 || stride < 1 || batch < 1) { \
        snprintf(last_error,sizeof(last_error),"invalid decode candidate dimensions"); return -1; \
    } \
    ACTIVATE; \
    switch(dim) { \
        case 32: return bs==256 ? launch<32,true>(DECODE_CALL) : launch<32,false>(DECODE_CALL); \
        case 64: return bs==256 ? launch<64,true>(DECODE_CALL) : launch<64,false>(DECODE_CALL); \
        case 128: return bs==256 ? launch<128,true>(DECODE_CALL) : launch<128,false>(DECODE_CALL); \
        case 256: return bs==256 ? launch<256,true>(DECODE_CALL) : launch<256,false>(DECODE_CALL); \
        default: snprintf(last_error,sizeof(last_error),"unsupported decode candidate head dimension"); return -1; \
    }
extern "C" int nv_attention_staged_candidate(DECODE_ARGS) { DECODE_DISPATCH(launch_decode_staged) }
extern "C" int nv_attention_splitkv_candidate(DECODE_ARGS) { DECODE_DISPATCH(launch_decode_splitkv) }
#undef DECODE_DISPATCH
#undef DECODE_CALL
#undef DECODE_ARGS

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstdio>
#include <new>

// The C ABI owns the CUDA stream/cuBLAS handle. Rust checks all buffer extents.
using bf16 = __nv_bfloat16;
struct Context { int device; cudaStream_t stream; cublasHandle_t blas; };
static thread_local char last_error[1024];
static int cuda_check(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return 0;
    snprintf(last_error, sizeof(last_error), "%s: %s (%d)", operation, cudaGetErrorString(status), int(status));
    return int(status);
}
static int blas_check(cublasStatus_t status, const char* operation) {
    if (status == CUBLAS_STATUS_SUCCESS) return 0;
    snprintf(last_error, sizeof(last_error), "%s: cuBLAS status %d", operation, int(status));
    return int(status) + 10000;
}
#define CUDA(call) do { int error = cuda_check((call), #call); if (error) return error; } while (0)
#define BLAS(call) do { int error = blas_check((call), #call); if (error) return error; } while (0)
#define ACTIVATE CUDA(cudaSetDevice(c->device))
#define LAUNCH_CHECK return cuda_check(cudaGetLastError(), "CUDA kernel launch")

extern "C" {
const char* nv_last_error() { return last_error; }
int nv_create(int device, Context** output) {
    CUDA(cudaSetDevice(device));
    Context* c = new (std::nothrow) Context{device, nullptr, nullptr};
    if (!c) { snprintf(last_error, sizeof(last_error), "host Context allocation failed"); return -1; }
    int error = cuda_check(cudaStreamCreateWithFlags(&c->stream, cudaStreamNonBlocking), "cudaStreamCreate");
    if (error) { delete c; return error; }
    error = blas_check(cublasCreate(&c->blas), "cublasCreate");
    if (error) { cudaStreamDestroy(c->stream); delete c; return error; }
    error = blas_check(cublasSetStream(c->blas, c->stream), "cublasSetStream");
    if (error) { cublasDestroy(c->blas); cudaStreamDestroy(c->stream); delete c; return error; }
    *output = c;
    return 0;
}
void nv_destroy(Context* c) {
    if (!c) return;
    cudaSetDevice(c->device);
    cudaStreamSynchronize(c->stream);
    cublasDestroy(c->blas);
    cudaStreamDestroy(c->stream);
    delete c;
}
int nv_alloc(Context* c, size_t bytes, void** output) {
    ACTIVATE;
    CUDA(cudaMalloc(output, bytes));
    int error = cuda_check(cudaMemsetAsync(*output, 0, bytes, c->stream), "cudaMemsetAsync");
    if (error) { cudaFree(*output); *output = nullptr; }
    return error;
}
void nv_free(Context* c, void* buffer) { cudaSetDevice(c->device); cudaFree(buffer); }
int nv_upload(Context* c, void* dst, const void* src, size_t bytes) {
    ACTIVATE;
    CUDA(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, c->stream));
    return cuda_check(cudaStreamSynchronize(c->stream), "host upload synchronize");
}
int nv_download(Context* c, const void* src, void* dst, size_t bytes) {
    ACTIVATE;
    CUDA(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, c->stream));
    return cuda_check(cudaStreamSynchronize(c->stream), "host download synchronize");
}
int nv_sync(Context* c) { ACTIVATE; return cuda_check(cudaStreamSynchronize(c->stream), "cudaStreamSynchronize"); }
int nv_mem_info(Context* c, size_t* free_bytes, size_t* total_bytes) { ACTIVATE; return cuda_check(cudaMemGetInfo(free_bytes, total_bytes), "cudaMemGetInfo"); }
int nv_capture_begin(Context* c) { ACTIVATE; return cuda_check(cudaStreamBeginCapture(c->stream, cudaStreamCaptureModeThreadLocal), "cudaStreamBeginCapture"); }
int nv_capture_end(Context* c, cudaGraphExec_t* output) {
    ACTIVATE;
    cudaGraph_t graph = nullptr;
    int error = cuda_check(cudaStreamEndCapture(c->stream, &graph), "cudaStreamEndCapture");
    if (error) { if (graph) cudaGraphDestroy(graph); return error; }
    error = cuda_check(cudaGraphInstantiate(output, graph, 0), "cudaGraphInstantiate");
    cudaGraphDestroy(graph);
    return error;
}
void nv_capture_abort(Context* c) {
    cudaSetDevice(c->device);
    cudaGraph_t graph = nullptr;
    cudaStreamEndCapture(c->stream, &graph);
    if (graph) cudaGraphDestroy(graph);
}
int nv_graph_replay(Context* c, cudaGraphExec_t graph) { ACTIVATE; return cuda_check(cudaGraphLaunch(graph, c->stream), "cudaGraphLaunch"); }
void nv_graph_destroy(Context* c, cudaGraphExec_t graph) { cudaSetDevice(c->device); cudaGraphExecDestroy(graph); }
int nv_gemm(Context* c, const bf16* x, const bf16* w, bf16* y, int m, int n, int k) {
    ACTIVATE;
    const float alpha = 1.0f, beta = 0.0f;
    // Row-major Y[M,N] = X[M,K] W[N,K]^T, expressed as Y^T = W X^T.
    BLAS(cublasGemmEx(c->blas, CUBLAS_OP_T, CUBLAS_OP_N, n, m, k,
        &alpha, w, CUDA_R_16BF, k, x, CUDA_R_16BF, k, &beta,
        y, CUDA_R_16BF, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    return 0;
}
}

__device__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_down_sync(0xffffffff, value, offset);
    return value;
}
__device__ float block_sum(float value) {
    __shared__ float partial[8];
    value = warp_sum(value);
    if ((threadIdx.x & 31) == 0) partial[threadIdx.x >> 5] = value;
    __syncthreads();
    if (threadIdx.x < 32) {
        value = threadIdx.x < 8 ? partial[threadIdx.x] : 0.0f;
        value = warp_sum(value);
        if (threadIdx.x == 0) partial[0] = value;
    }
    __syncthreads();
    return partial[0];
}
__device__ float block_max(float value) {
    __shared__ float partial[8];
    for (int offset = 16; offset > 0; offset >>= 1) value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    if ((threadIdx.x & 31) == 0) partial[threadIdx.x >> 5] = value;
    __syncthreads();
    if (threadIdx.x < 32) {
        value = threadIdx.x < 8 ? partial[threadIdx.x] : -INFINITY;
        for (int offset = 16; offset > 0; offset >>= 1) value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
        if (threadIdx.x == 0) partial[0] = value;
    }
    __syncthreads();
    return partial[0];
}
__global__ void embedding_kernel(const int* tokens, const bf16* weight, bf16* out, int rows, int hidden, int vocab) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < int64_t(rows) * hidden; i += int64_t(gridDim.x) * blockDim.x) {
        int token = tokens[i / hidden];
        out[i] = token >= 0 && token < vocab ? weight[int64_t(token) * hidden + i % hidden] : __float2bfloat16(0.0f);
    }
}
template<bool ADD>
__global__ void rms_kernel(const bf16* x, bf16* residual, const bf16* weight, bf16* out, int hidden, float eps) {
    int64_t base = int64_t(blockIdx.x) * hidden;
    float square_sum = 0;
    for (int64_t j = threadIdx.x; j < hidden; j += blockDim.x) {
        float value = __bfloat162float(x[base + j]);
        if (ADD) value += __bfloat162float(residual[base + j]);
        square_sum += value * value;
    }
    float inv_rms = rsqrtf(block_sum(square_sum) / hidden + eps);
    for (int64_t j = threadIdx.x; j < hidden; j += blockDim.x) {
        float value = __bfloat162float(x[base + j]);
        if (ADD) { value += __bfloat162float(residual[base + j]); residual[base + j] = __float2bfloat16_rn(value); }
        // nano-vLLM compiles RMSNorm with torch.compile. Inductor's default
        // emulate_precision_casts=false elides this intermediate BF16 cast;
        // only the norm module's stored output rounds to BF16.
        float normalized = value * inv_rms;
        out[base + j] = __float2bfloat16_rn(normalized * __bfloat162float(weight[j]));
    }
}

template<bool CACHED>
__global__ void qkv_rope_kernel(const bf16* qkv, const bf16* qweight, const bf16* kweight,
    const int* positions, const int* slots, bf16* qout, bf16* kcache, bf16* vcache,
    int qheads, int kvheads, int dim, int block_size, int num_blocks, float eps, float theta,
    const float* rope_cache, int max_positions) {
    const int row = blockIdx.x, head = blockIdx.y;
    const bool is_q = head < qheads;
    const int local_head = is_q ? head : head - qheads;
    const int64_t packed = int64_t(row) * (qheads + 2 * kvheads) * dim;
    const int64_t base = packed + int64_t(head) * dim;
    const bf16* weight = is_q ? qweight : kweight;
    __shared__ float normalized[256];
    float sum = 0;
    for (int j = threadIdx.x; j < dim; j += blockDim.x) { float v = __bfloat162float(qkv[base + j]); sum += v * v; }
    float inv_rms = rsqrtf(block_sum(sum) / dim + eps);
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        // Keep normalization in FP32 as in Inductor's fused RMSNorm, then
        // materialize the BF16 norm-module output before the separate RoPE.
        float norm = __bfloat162float(qkv[base + j]) * inv_rms;
        normalized[j] = __bfloat162float(__float2bfloat16_rn(norm * __bfloat162float(weight[j])));
    }
    __syncthreads();
    const int position = positions[row], slot = slots[row];
    const bool valid_slot = slot >= 0 && int64_t(slot) < int64_t(block_size) * num_blocks;
    const int64_t cache_base = (int64_t(slot) * kvheads + local_head) * dim;
    for (int j = threadIdx.x; j < dim; j += blockDim.x) {
        int half = dim / 2, pair = j % half;
        float sine, cosine;
        if constexpr (CACHED) {
            bool valid_position = position >= 0 && position < max_positions;
            cosine = valid_position ? rope_cache[int64_t(position) * dim + pair] : 1.0f;
            sine = valid_position ? rope_cache[int64_t(position) * dim + half + pair] : 0.0f;
        } else {
            float angle = position * powf(theta, -float(2 * pair) / dim);
            sincosf(angle, &sine, &cosine);
        }
        // Disable contraction here to match the two products in reference RoPE.
        float rotated = j < half ? __fsub_rn(__fmul_rn(normalized[j], cosine), __fmul_rn(normalized[j + half], sine))
                                 : __fadd_rn(__fmul_rn(normalized[j], cosine), __fmul_rn(normalized[j - half], sine));
        if (is_q) qout[(int64_t(row) * qheads + local_head) * dim + j] = __float2bfloat16_rn(rotated);
        else if (valid_slot) {
            kcache[cache_base + j] = __float2bfloat16_rn(rotated);
            vcache[cache_base + j] = qkv[packed + int64_t(qheads + kvheads + local_head) * dim + j];
        }
    }
}

__global__ void rope_cache_kernel(float* cache, int positions, int dim, float theta) {
    const int half = dim / 2;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < int64_t(positions) * half; i += int64_t(gridDim.x) * blockDim.x) {
        int position = int(i / half), pair = int(i % half);
        float angle = position * powf(theta, -float(2 * pair) / dim);
        float sine, cosine;
        sincosf(angle, &sine, &cosine);
        cache[int64_t(position) * dim + pair] = cosine;
        cache[int64_t(position) * dim + half + pair] = sine;
    }
}

__global__ void attention_kernel(const bf16* q, const bf16* kcache, const bf16* vcache,
    const int* positions, const int* seq_indices, const int* tables, bf16* out,
    int qheads, int kvheads, int dim, int block_size, int num_blocks, int stride, int batch, int max_context) {
    int row = blockIdx.x, head = blockIdx.y, kvhead = head / (qheads / kvheads);
    int seq = seq_indices[row];
    int64_t context64 = int64_t(positions[row]) + 1;
    int64_t qbase = (int64_t(row) * qheads + head) * dim;
    if (seq < 0 || seq >= batch || context64 < 1 || context64 > max_context || context64 > int64_t(stride) * block_size) {
        for (int d = threadIdx.x; d < dim; d += blockDim.x) out[qbase + d] = __float2bfloat16(0.0f);
        return;
    }
    int context = int(context64);
    extern __shared__ float scores[];
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    float max_score = -INFINITY;
    for (int token = warp; token < context; token += 8) {
        int block = tables[int64_t(seq) * stride + token / block_size];
        float dot = 0;
        if (block >= 0 && block < num_blocks) {
            int64_t base = ((int64_t(block) * block_size + token % block_size) * kvheads + kvhead) * dim;
            for (int d = lane; d < dim; d += 32) dot += __bfloat162float(q[qbase + d]) * __bfloat162float(kcache[base + d]);
            dot = warp_sum(dot) * rsqrtf(float(dim));
        } else dot = -INFINITY;
        if (lane == 0) { scores[token] = dot; max_score = fmaxf(max_score, dot); }
    }
    max_score = block_max(max_score);
    float sum = 0;
    for (int token = threadIdx.x; token < context; token += blockDim.x) {
        float value = isfinite(max_score) ? expf(scores[token] - max_score) : 0.0f;
        scores[token] = value;
        sum += value;
    }
    float denominator = block_sum(sum);
    // All dimensions in a warp access consecutive V elements (coalesced).
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float value = 0;
        for (int token = 0; token < context; ++token) {
            int block = tables[int64_t(seq) * stride + token / block_size];
            if (block >= 0 && block < num_blocks) {
                int64_t base = ((int64_t(block) * block_size + token % block_size) * kvheads + kvhead) * dim;
                value += scores[token] * __bfloat162float(vcache[base + d]);
            }
        }
        out[qbase + d] = __float2bfloat16_rn(denominator > 0 ? value / denominator : 0.0f);
    }
}

// The common head dimensions have fixed register arrays and fully unrolled
// inner dot products. BLOCK256 removes integer division from the hot KV loop.
template<bool BLOCK256>
__device__ __forceinline__ int64_t kv_token_base(int token, const int* table,
    int block_size, int num_blocks, int kvheads, int kvhead, int dim) {
    int logical, offset;
    if constexpr (BLOCK256) { logical = token >> 8; offset = token & 255; }
    else { logical = token / block_size; offset = token % block_size; }
    const int block = table[logical];
    if (block < 0 || block >= num_blocks) return -1;
    return ((int64_t(block) * block_size + offset) * kvheads + kvhead) * dim;
}

// Optimized attention preserves the baseline FP32 accumulation order.
// Block-wise address hoisting and register prefetch change scheduling, not rounding.
template<int DIM, bool BLOCK256>
__global__ void attention_ordered_kernel(const bf16* q, const bf16* kcache, const bf16* vcache,
    const int* positions, const int* seq_indices, const int* tables, bf16* out,
    int qheads, int kvheads, int block_size, int num_blocks, int stride, int batch, int max_context, int runtime_dim) {
    const int row=blockIdx.x, head=blockIdx.y, seq=seq_indices[row], kvhead=head/(qheads/kvheads);
    const int64_t context64=int64_t(positions[row])+1, qbase=(int64_t(row)*qheads+head)*DIM;
    if(seq<0 || seq>=batch || context64<1 || context64>max_context || context64>int64_t(stride)*block_size) {
        if(threadIdx.x<DIM) out[qbase+threadIdx.x]=__float2bfloat16(0.0f);
        return;
    }
    const int context=int(context64), warp=threadIdx.x>>5, lane=threadIdx.x&31;
    const int* table=tables+int64_t(seq)*stride;
    extern __shared__ float scores[];
    float query[DIM/32];
    #pragma unroll
    for(int d=0;d<DIM/32;++d) query[d]=__bfloat162float(q[qbase+lane+32*d]);
    float max_score=-INFINITY;
    for(int token=warp;token<context;token+=8) {
        int64_t base=kv_token_base<BLOCK256>(token,table,block_size,num_blocks,kvheads,kvhead,DIM);
        float dot=0;
        if(base>=0) {
            #pragma unroll
            for(int d=0;d<DIM/32;++d) dot+=query[d]*__bfloat162float(kcache[base+lane+32*d]);
            // Keep runtime rsqrt: nvcc's folded rsqrt(128) is one ULP above
            // the GPU rsqrt instruction used by the baseline.
            dot=warp_sum(dot)*rsqrtf(float(runtime_dim));
        } else dot=-INFINITY;
        if(lane==0) {scores[token]=dot;max_score=fmaxf(max_score,dot);}
    }
    max_score=block_max(max_score);
    float sum=0;
    for(int token=threadIdx.x;token<context;token+=blockDim.x) {
        float probability=isfinite(max_score)?expf(scores[token]-max_score):0.0f;
        scores[token]=probability;sum+=probability;
    }
    const float denominator=block_sum(sum);
    if(threadIdx.x<DIM) {
        float value=0;
        const int size=BLOCK256?256:block_size;
        // Resolve each physical block once. Within a block addresses progress
        // with a fixed stride, so loads no longer depend on a per-token table read.
        for(int start=0,logical=0;start<context;start+=size,++logical) {
            const int block=table[logical],count=min(size,context-start);
            if(block<0 || block>=num_blocks) continue;
            const bf16* base=vcache+(int64_t(block)*block_size*kvheads+kvhead)*DIM+threadIdx.x;
            for(int offset=0;offset<count;offset+=8) {
                float probability[8],values[8];
                #pragma unroll
                for(int j=0;j<8;++j) if(offset+j<count) {
                    probability[j]=scores[start+offset+j];
                    values[j]=__bfloat162float(base[int64_t(offset+j)*kvheads*DIM]);
                }
                // Unrolling schedules independent loads together, but these
                // additions still run strictly token 0,1,2,... in baseline order.
                #pragma unroll
                for(int j=0;j<8;++j) if(offset+j<count) value+=probability[j]*values[j];
            }
        }
        out[qbase+threadIdx.x]=__float2bfloat16_rn(denominator>0?value/denominator:0.0f);
    }
}
template<int DIM>
static void launch_ordered_attention(Context* c,const bf16* q,const bf16* k,const bf16* v,
    const int* p,const int* s,const int* t,bf16* o,int rows,int qh,int kvh,int bs,int blocks,int stride,int batch,int max_context) {
    if(bs==256) attention_ordered_kernel<DIM,true><<<dim3(rows,qh),256,size_t(max_context)*sizeof(float),c->stream>>>(q,k,v,p,s,t,o,qh,kvh,bs,blocks,stride,batch,max_context,DIM);
    else attention_ordered_kernel<DIM,false><<<dim3(rows,qh),256,size_t(max_context)*sizeof(float),c->stream>>>(q,k,v,p,s,t,o,qh,kvh,bs,blocks,stride,batch,max_context,DIM);
}

__global__ void silu_kernel(const bf16* packed, bf16* out, int rows, int intermediate) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < int64_t(rows) * intermediate; i += int64_t(gridDim.x) * blockDim.x) {
        int64_t base = (i / intermediate) * (2 * intermediate) + i % intermediate;
        float gate = __bfloat162float(packed[base]);
        // torch.compile fuses SiluAndMul and elides SiLU's intermediate BF16
        // round-trip. Match that FP32 expression with one final BF16 store.
        float activation = gate / (1.0f + expf(-gate));
        out[i] = __float2bfloat16_rn(activation * __bfloat162float(packed[base + intermediate]));
    }
}
__global__ void gather_kernel(const bf16* x, const int* indices, bf16* out, int rows, int hidden, int input_rows) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < int64_t(rows) * hidden; i += int64_t(gridDim.x) * blockDim.x) {
        int index = indices[i / hidden];
        out[i] = index >= 0 && index < input_rows ? x[int64_t(index) * hidden + i % hidden] : __float2bfloat16(0.0f);
    }
}
__device__ uint64_t splitmix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}
__global__ void sample_kernel(const bf16* logits, const float* temperatures, const uint64_t* seeds, int* out, int vocab) {
    int row = blockIdx.x;
    float temperature = temperatures[row], best = -INFINITY;
    int best_index = 0x7fffffff;
    for (int64_t token = threadIdx.x; token < vocab; token += blockDim.x) {
        float score = __bfloat162float(logits[int64_t(row) * vocab + token]);
        if (temperature > 0) {
            uint64_t random = splitmix64(seeds[row] ^ (uint64_t(token) * 0xd2b74407b1ce6e93ULL));
            float uniform = (float(random >> 41) + 0.5f) * (1.0f / 8388608.0f);
            score = score / temperature - logf(-logf(uniform));
        }
        if (score > best || (score == best && token < best_index)) { best = score; best_index = int(token); }
    }
    __shared__ float best_scores[256];
    __shared__ int best_indices[256];
    best_scores[threadIdx.x] = best;
    best_indices[threadIdx.x] = best_index;
    __syncthreads();
    for (int offset = 128; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            float candidate = best_scores[threadIdx.x + offset];
            int index = best_indices[threadIdx.x + offset];
            if (candidate > best_scores[threadIdx.x] || (candidate == best_scores[threadIdx.x] && index < best_indices[threadIdx.x])) {
                best_scores[threadIdx.x] = candidate; best_indices[threadIdx.x] = index;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[row] = best_indices[0] == 0x7fffffff ? 0 : best_indices[0];
}

// A two-level argmax exposes vocabulary parallelism even for batch size one.
// SplitMix64 and Gumbel scores are unchanged; score/index reductions choose the
// same lowest token index on ties, independent of partitioning.
__device__ __forceinline__ void warp_argmax(float& score, int& index) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const float other = __shfl_down_sync(0xffffffff, score, offset);
        const int other_index = __shfl_down_sync(0xffffffff, index, offset);
        if (other > score || (other == score && other_index < index)) { score = other; index = other_index; }
    }
}
__device__ void block_argmax(float& score, int& index) {
    __shared__ float scores[8];
    __shared__ int indices[8];
    warp_argmax(score,index);
    if ((threadIdx.x & 31) == 0) { scores[threadIdx.x >> 5] = score; indices[threadIdx.x >> 5] = index; }
    __syncthreads();
    if (threadIdx.x < 32) {
        score = threadIdx.x < 8 ? scores[threadIdx.x] : -INFINITY;
        index = threadIdx.x < 8 ? indices[threadIdx.x] : 0x7fffffff;
        warp_argmax(score,index);
    }
}
constexpr int SAMPLE_PARTITION_SIZE = 2048;
__global__ void sample_partition_kernel(const bf16* logits, const float* temperatures,
    const uint64_t* seeds, float* partial_scores, int* partial_indices, int vocab, int partitions) {
    const int row = blockIdx.x / partitions, part = blockIdx.x % partitions;
    const int64_t begin = int64_t(part) * SAMPLE_PARTITION_SIZE;
    const int64_t end = min(begin + SAMPLE_PARTITION_SIZE, int64_t(vocab));
    const float temperature = temperatures[row];
    const uint64_t seed = seeds[row];
    float best = -INFINITY;
    int best_index = 0x7fffffff;
    for (int64_t token = begin + threadIdx.x; token < end; token += blockDim.x) {
        float score = __bfloat162float(logits[int64_t(row)*vocab + token]);
        if (temperature > 0) {
            uint64_t random = splitmix64(seed ^ (uint64_t(token) * 0xd2b74407b1ce6e93ULL));
            float uniform = (float(random >> 41) + 0.5f) * (1.0f / 8388608.0f);
            score = score / temperature - logf(-logf(uniform));
        }
        if (score > best || (score == best && token < best_index)) { best = score; best_index = int(token); }
    }
    block_argmax(best,best_index);
    if (threadIdx.x == 0) { partial_scores[blockIdx.x] = best; partial_indices[blockIdx.x] = best_index; }
}
__global__ void sample_combine_kernel(const float* partial_scores, const int* partial_indices,
    int* out, int partitions) {
    const int64_t base = int64_t(blockIdx.x) * partitions;
    float best = -INFINITY;
    int best_index = 0x7fffffff;
    for (int64_t p = threadIdx.x; p < partitions; p += blockDim.x) {
        const float score = partial_scores[base+p];
        const int index = partial_indices[base+p];
        if (score > best || (score == best && index < best_index)) { best = score; best_index = index; }
    }
    block_argmax(best,best_index);
    if (threadIdx.x == 0) out[blockIdx.x] = best_index == 0x7fffffff ? 0 : best_index;
}
static int grid(int64_t elements) { return int(elements / 256 + (elements % 256 != 0) > 65535 ? 65535 : (elements + 255) / 256); }
extern "C" {
int nv_embedding(Context* c, const int* tokens, const bf16* weight, bf16* out, int rows, int hidden, int vocab) { ACTIVATE; embedding_kernel<<<grid(int64_t(rows)*hidden),256,0,c->stream>>>(tokens,weight,out,rows,hidden,vocab); LAUNCH_CHECK; }
int nv_rms(Context* c, const bf16* x, const bf16* weight, bf16* out, int rows, int hidden, float eps) { ACTIVATE; rms_kernel<false><<<rows,256,0,c->stream>>>(x,nullptr,weight,out,hidden,eps); LAUNCH_CHECK; }
int nv_add_rms(Context* c, const bf16* x, bf16* residual, const bf16* weight, bf16* out, int rows, int hidden, float eps) { ACTIVATE; rms_kernel<true><<<rows,256,0,c->stream>>>(x,residual,weight,out,hidden,eps); LAUNCH_CHECK; }
int nv_qkv_rope(Context* c, const bf16* qkv, const bf16* qw, const bf16* kw, const int* positions, const int* slots, bf16* qout, bf16* kc, bf16* vc, int rows, int qh, int kvh, int dim, int bs, int blocks, float eps, float theta) { ACTIVATE; qkv_rope_kernel<false><<<dim3(rows,qh+kvh),256,0,c->stream>>>(qkv,qw,kw,positions,slots,qout,kc,vc,qh,kvh,dim,bs,blocks,eps,theta,nullptr,0); LAUNCH_CHECK; }
int nv_rope_cache(Context* c, float* cache, int positions, int dim, float theta) { ACTIVATE; rope_cache_kernel<<<grid(int64_t(positions)*(dim/2)),256,0,c->stream>>>(cache,positions,dim,theta); LAUNCH_CHECK; }
int nv_qkv_rope_cached(Context* c, const bf16* qkv, const bf16* qw, const bf16* kw, const int* positions, const int* slots, bf16* qout, bf16* kc, bf16* vc, int rows, int qh, int kvh, int dim, int bs, int blocks, float eps, const float* rope_cache, int max_positions) { ACTIVATE; qkv_rope_kernel<true><<<dim3(rows,qh+kvh),256,0,c->stream>>>(qkv,qw,kw,positions,slots,qout,kc,vc,qh,kvh,dim,bs,blocks,eps,1.0f,rope_cache,max_positions); LAUNCH_CHECK; }
int nv_attention(Context* c, const bf16* q, const bf16* kc, const bf16* vc, const int* positions, const int* seqs, const int* tables, bf16* out, int rows, int qh, int kvh, int dim, int bs, int blocks, int stride, int batch, int max_context) {
    ACTIVATE;
    switch (dim) {
        case 32: launch_ordered_attention<32>(c,q,kc,vc,positions,seqs,tables,out,rows,qh,kvh,bs,blocks,stride,batch,max_context); break;
        case 64: launch_ordered_attention<64>(c,q,kc,vc,positions,seqs,tables,out,rows,qh,kvh,bs,blocks,stride,batch,max_context); break;
        case 128: launch_ordered_attention<128>(c,q,kc,vc,positions,seqs,tables,out,rows,qh,kvh,bs,blocks,stride,batch,max_context); break;
        case 256: launch_ordered_attention<256>(c,q,kc,vc,positions,seqs,tables,out,rows,qh,kvh,bs,blocks,stride,batch,max_context); break;
        default: attention_kernel<<<dim3(rows,qh),256,size_t(max_context)*sizeof(float),c->stream>>>(q,kc,vc,positions,seqs,tables,out,qh,kvh,dim,bs,blocks,stride,batch,max_context);
    }
    LAUNCH_CHECK;
}
int nv_silu(Context* c, const bf16* packed, bf16* out, int rows, int intermediate) { ACTIVATE; silu_kernel<<<grid(int64_t(rows)*intermediate),256,0,c->stream>>>(packed,out,rows,intermediate); LAUNCH_CHECK; }
int nv_gather(Context* c, const bf16* x, const int* indices, bf16* out, int rows, int hidden, int input_rows) { ACTIVATE; gather_kernel<<<grid(int64_t(rows)*hidden),256,0,c->stream>>>(x,indices,out,rows,hidden,input_rows); LAUNCH_CHECK; }
int nv_sample(Context* c, const bf16* logits, const float* temps, const uint64_t* seeds, int* out, int batch, int vocab) { ACTIVATE; sample_kernel<<<batch,256,0,c->stream>>>(logits,temps,seeds,out,vocab); LAUNCH_CHECK; }
int nv_sample_parallel(Context* c, const bf16* logits, const float* temps, const uint64_t* seeds, int* out, float* partial_scores, int* partial_indices, int batch, int vocab) {
    ACTIVATE;
    const int partitions = int((int64_t(vocab) + SAMPLE_PARTITION_SIZE - 1) / SAMPLE_PARTITION_SIZE);
    const int64_t launches = int64_t(batch) * partitions;
    if (partitions <= 1 || launches > 0x7fffffff) {
        sample_kernel<<<batch,256,0,c->stream>>>(logits,temps,seeds,out,vocab);
    } else {
        sample_partition_kernel<<<int(launches),256,0,c->stream>>>(logits,temps,seeds,partial_scores,partial_indices,vocab,partitions);
        CUDA(cudaGetLastError());
        sample_combine_kernel<<<batch,256,0,c->stream>>>(partial_scores,partial_indices,out,partitions);
    }
    LAUNCH_CHECK;
}
}

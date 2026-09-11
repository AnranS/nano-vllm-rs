// Standalone numerical integration test; compile with the documented command.
#include "flash_native.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

static void check(cudaError_t e) { if (e) throw std::runtime_error(cudaGetErrorString(e)); }
template<class T> struct Device {
    T* p = nullptr;
    explicit Device(size_t count) { check(cudaMalloc(&p, count * sizeof(T))); }
    ~Device() { cudaFree(p); }
    void upload(const std::vector<T>& v) { check(cudaMemcpy(p, v.data(), v.size()*sizeof(T), cudaMemcpyHostToDevice)); }
};

static void run_case(const char* name, std::vector<int> qlens,
                     std::vector<int> klens, int splits, bool graph,
                     int h=16, int hk=8) {
    constexpr int d=128, page=256, stride=4, pages=12;
    const int batch=int(qlens.size()), maxq=*std::max_element(qlens.begin(), qlens.end());
    std::vector<int> cu(batch+1);
    for(int b=0;b<batch;++b) cu[b+1]=cu[b]+qlens[b];
    std::vector<int> table{7,2,10,1, 3,8,0,6, 11,5,9,4};
    std::vector<__nv_bfloat16> q(size_t(cu.back())*h*d), k(size_t(pages)*page*hk*d), v(k.size());
    for(size_t i=0;i<q.size();++i) q[i]=__float2bfloat16(std::sin(float(i%1103)*.27f)*.4f);
    for(size_t i=0;i<k.size();++i) {
        k[i]=__float2bfloat16(std::cos(float(i%1091)*.19f)*.3f);
        v[i]=__float2bfloat16(std::sin(float(i%907)*.07f)*.6f);
    }
    Device<__nv_bfloat16> dq(q.size()), dk(k.size()), dv(v.size()), out(q.size());
    Device<int> dc(cu.size()), dl(klens.size()), dt(table.size());
    Device<float> lse(size_t(batch)*h*maxq), la(size_t(splits)*batch*h), oa(size_t(splits)*batch*h*d);
    dq.upload(q);dk.upload(k);dv.upload(v);dc.upload(cu);dl.upload(klens);dt.upload(table);
    cudaStream_t stream; check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    auto launch=[&]{
        if(nvr_flash_fwd(stream,dq.p,dk.p,dv.p,out.p,dc.p,dl.p,dt.p,
                         batch,cu.back(),maxq,1024,h,hk,page,stride,splits,lse.p,la.p,oa.p))
            throw std::runtime_error(nvr_flash_last_error());
    };
    launch(); check(cudaStreamSynchronize(stream));
    if(graph) {
        cudaGraph_t graph_handle;cudaGraphExec_t graph_exec;
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));launch();
        check(cudaStreamEndCapture(stream,&graph_handle));
        check(cudaGraphInstantiate(&graph_exec,graph_handle,nullptr,nullptr,0));
        klens={40,450,700};dl.upload(klens);
        check(cudaGraphLaunch(graph_exec,stream));check(cudaStreamSynchronize(stream));
        check(cudaGraphExecDestroy(graph_exec));check(cudaGraphDestroy(graph_handle));
    }
    std::vector<__nv_bfloat16> result(q.size());
    check(cudaMemcpy(result.data(),out.p,result.size()*sizeof(__nv_bfloat16),cudaMemcpyDeviceToHost));
    double max_error=0,sum_error=0;size_t count=0;
    for(int b=0;b<batch;++b) for(int row=0;row<qlens[b];++row) for(int head=0;head<h;++head) {
        const int n=klens[b]-qlens[b]+row+1;
        const int kh=head/(h/hk);
        std::vector<float> scores(n);float m=-INFINITY;
        for(int j=0;j<n;++j) {
            size_t kbase=(size_t(table[b*stride+j/page])*page+j%page)*hk*d+kh*d;
            size_t qbase=(size_t(cu[b]+row)*h+head)*d;
            float dot=0;
            for(int x=0;x<d;++x) dot+=__bfloat162float(q[qbase+x])*__bfloat162float(k[kbase+x]);
            scores[j]=dot/std::sqrt(float(d));m=std::max(m,scores[j]);
        }
        float total=0;for(float& s:scores){s=std::exp(s-m);total+=s;}
        for(int x=0;x<d;++x){
            float val=0;
            for(int j=0;j<n;++j){
                size_t base=(size_t(table[b*stride+j/page])*page+j%page)*hk*d+kh*d;
                val+=scores[j]*__bfloat162float(v[base+x]);
            }
            float expected=val/total;
            float actual=__bfloat162float(result[(size_t(cu[b]+row)*h+head)*d+x]);
            const double error=std::abs(actual-expected);
            if(!std::isfinite(actual))throw std::runtime_error("nonfinite output");
            max_error=std::max(max_error,error);sum_error+=error;++count;
        }
    }
    check(cudaStreamDestroy(stream));
    std::printf("%s splits=%d graph=%d max_abs_error=%.8g mean_abs_error=%.8g\n",name,splits,graph,max_error,sum_error/count);
    if(max_error>0.003)throw std::runtime_error("attention differs from independent FP32 reference");
}

int main(){try{
    run_case("ragged_paged_causal_prefill",{17,65,7},{33,400,270},1,false);
    for(int splits:{1,2,8,32})run_case("paged_gqa_decode",{1,1,1},{33,400,1023},splits,false);
    run_case("paged_mha_decode",{1,1,1},{33,400,1023},8,false,8,8);
    run_case("paged_mqa_decode",{1,1,1},{33,400,1023},8,false,16,1);
    run_case("paged_gqa_decode",{1,1,1},{33,400,1023},8,true);
    std::puts("All standalone FlashAttention numerical checks passed.");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}

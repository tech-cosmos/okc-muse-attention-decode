"""Native CUDA experiment. Development harness imports this; not a submitted dependency."""
import torch
from torch.utils.cpp_extension import load_inline

CPP = r'''
#include <torch/extension.h>
void attention(torch::Tensor qkv, torch::Tensor kc, torch::Tensor vc, torch::Tensor pt,
 torch::Tensor seq, torch::Tensor loc, torch::Tensor ks, torch::Tensor vs,
 torch::Tensor pa, torch::Tensor pm, int splits, int page, double scale, int stages);
'''
CUDA = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>
using namespace nvcuda;
__device__ __forceinline__ float warp_sum(float v) {
 #pragma unroll
 for(int d=16; d; d/=2) v+=__shfl_xor_sync(0xffffffff,v,d);
 return v;
}
__device__ __forceinline__ half from8(unsigned char v) {
 return __nv_cvt_fp8_to_halfraw(v,__NV_E4M3);
}
__device__ __forceinline__ unsigned char to8(float v) {
 return __nv_cvt_float_to_fp8(v,__NV_SATFINITE,__NV_E4M3);
}
__device__ __forceinline__ void copy16(void* dst, const void* src) {
 unsigned sm=__cvta_generic_to_shared(dst);
 asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"(sm),"l"(src));
}
__device__ __forceinline__ void consumer_barrier() {
 asm volatile("bar.sync 1, 128;":::"memory");
}
__device__ __forceinline__ unsigned acquire(unsigned* p) {
 unsigned v; asm volatile("ld.acquire.cta.shared.u32 %0, [%1];":"=r"(v):"r"((unsigned)__cvta_generic_to_shared(p)):"memory"); return v;
}
__device__ __forceinline__ void release(unsigned* p,unsigned v) {
 asm volatile("st.release.cta.shared.u32 [%0], %1;"::"r"((unsigned)__cvta_generic_to_shared(p)),"r"(v):"memory");
}

template<int STAGES>
__global__ void verify(const __nv_bfloat16* __restrict__ qkv, unsigned char* kc, unsigned char* vc,
 const int* __restrict__ pt, const int* __restrict__ seqs, const long* __restrict__ loc,
 const float* ks, const float* vs, float* __restrict__ pa, float* __restrict__ pm,
 int HQ,int HKV,int MP,int PAGE,int SPLITS,float SCALE) {
 constexpr int N=32, M=64, D=64;
 __shared__ __align__(16) unsigned char raw[STAGES][2][N*D];
 __shared__ __align__(16) unsigned char own[2][8*D];
 __shared__ __align__(32) half Q[M*D], K[N*D], V[N*D], P[M*N];
 __shared__ unsigned ready[STAGES];
 const int tid=threadIdx.x, warp=tid/32, lane=tid%32;
 const int b=blockIdx.y, h=blockIdx.z, split=blockIdx.x;
 const int seq=seqs[b], width=(2*HQ+2*HKV)*D, G=HQ/HKV;
 const int chunk=((seq+N*SPLITS-1)/(N*SPLITS))*N;
 const int lo=split*chunk, hi=min(seq,lo+chunk);
 const int tiles=max(0,(hi-lo+N-1)/N)+(split==0);
 const float kscale=*ks, vscale=*vs;
 for(int i=tid;i<STAGES;i+=blockDim.x) ready[i]=0;
 // Each warp normalizes one row at a time, exactly the same query FP8 rounding as FA3.
 for(int row=warp;row<M;row+=5) {
   int tok=b*8+row/G, head=h*G+row%G;
   float x=__bfloat162float(qkv[tok*width+head*D+lane]);
   float y=__bfloat162float(qkv[tok*width+head*D+lane+32]);
   float r=rsqrtf(warp_sum(x*x+y*y)/64.f+1e-5f);
   Q[row*D+lane]=from8(to8(x*r)); Q[row*D+lane+32]=from8(to8(y*r));
 }
 // New keys are handled without depending on concurrent writes to the shared cache.
 if(split==0) for(int row=warp;row<8;row+=5) {
   int tok=b*8+row;
   float x=__bfloat162float(qkv[tok*width+(HQ+h)*D+lane]);
   float y=__bfloat162float(qkv[tok*width+(HQ+h)*D+lane+32]);
   float r=rsqrtf(warp_sum(x*x+y*y)/64.f+1e-5f)/kscale;
   unsigned char k0=to8(x*r), k1=to8(y*r);
   unsigned char v0=to8(__bfloat162float(qkv[tok*width+(HQ+HKV+h)*D+lane])/vscale);
   unsigned char v1=to8(__bfloat162float(qkv[tok*width+(HQ+HKV+h)*D+lane+32])/vscale);
   own[0][row*D+lane]=k0; own[0][row*D+lane+32]=k1;
   own[1][row*D+lane]=v0; own[1][row*D+lane+32]=v1;
   long slot=loc[tok]*HKV*D+h*D;
   kc[slot+lane]=k0; kc[slot+lane+32]=k1;
   vc[slot+lane]=v0; vc[slot+lane+32]=v1;
 }
 __syncthreads();
 if(warp==4) {
   for(int tile=0;tile<tiles;tile++) {
     int stage=tile%STAGES;
     while(acquire(ready+stage)!=0) {}
     bool isnew=split==0 && tile==tiles-1;
     if(!isnew) {
       for(int i=lane;i<N*4;i+=32) {
         int t=min(lo+tile*N+i/4,hi-1);
         long row=((long)pt[b*MP+t/PAGE]*PAGE+t%PAGE)*HKV*D+h*D;
         copy16(raw[stage][0]+i*16,kc+row+(i%4)*16);
         copy16(raw[stage][1]+i*16,vc+row+(i%4)*16);
       }
       asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;":::"memory");
     } else {
       for(int i=lane;i<N*D;i+=32) {
         raw[stage][0][i]=i<8*D?own[0][i]:0;
         raw[stage][1][i]=i<8*D?own[1][i]:0;
       }
     }
     __syncwarp();
     if(lane==0) release(ready+stage,1);
   }
   return;
 }
 wmma::fragment<wmma::accumulator,16,16,16,float> acc[4];
 #pragma unroll
 for(int n=0;n<4;n++) wmma::fill_fragment(acc[n],0.f);
 float mx[2]={-1e30f,-1e30f}, sum[2]={0,0};
 wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> q[4];
 #pragma unroll
 for(int d=0;d<4;d++) wmma::load_matrix_sync(q[d],Q+warp*16*D+d*16,D);
 for(int tile=0;tile<tiles;tile++) {
   int stage=tile%STAGES;
   while(acquire(ready+stage)!=1) {}
   for(int i=tid;i<N*D;i+=128) {
     K[i]=from8(raw[stage][0][i]); V[i]=from8(raw[stage][1][i]);
   }
   consumer_barrier();
   if(tid==0) release(ready+stage,0);
   wmma::fragment<wmma::accumulator,16,16,16,float> score[2];
   #pragma unroll
   for(int n=0;n<2;n++) {
     wmma::fill_fragment(score[n],0.f);
     #pragma unroll
     for(int d=0;d<4;d++) {
       wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> k;
       wmma::load_matrix_sync(k,K+n*16*D+d*16,D);
       wmma::mma_sync(score[n],q[d],k,score[n]);
     }
   }
   bool isnew=split==0 && tile==tiles-1;
   float now[2]={-1e30f,-1e30f};
   #pragma unroll
   for(int n=0;n<2;n++) {
     #pragma unroll
     for(int i=0;i<8;i++) {
       int r=(i&2)/2, row=warp*16+lane/4+r*8;
       int col=n*16+(lane%4)*2+(i&1)+(i&4)*2;
       bool ok=isnew ? col<=row/G : lo+tile*N+col<hi;
       float s=ok?score[n].x[i]*SCALE*kscale:-1e30f;
       score[n].x[i]=s; now[r]=fmaxf(now[r],s);
     }
   }
   float alpha[2];
   #pragma unroll
   for(int r=0;r<2;r++) {
     now[r]=fmaxf(now[r],__shfl_xor_sync(0xffffffff,now[r],1));
     now[r]=fmaxf(now[r],__shfl_xor_sync(0xffffffff,now[r],2));
     now[r]=fmaxf(now[r],mx[r]); alpha[r]=exp2f(mx[r]-now[r]);
     sum[r]*=alpha[r]; mx[r]=now[r];
   }
   float add[2]={0,0};
   #pragma unroll
   for(int n=0;n<2;n++) {
     #pragma unroll
     for(int i=0;i<8;i++) {
       int r=(i&2)/2, row=warp*16+lane/4+r*8;
       int col=n*16+(lane%4)*2+(i&1)+(i&4)*2;
       float p=exp2f(score[n].x[i]-mx[r]);
       add[r]+=p; P[row*N+col]=__float2half_rn(p);
     }
   }
   #pragma unroll
   for(int r=0;r<2;r++) {
     add[r]+=__shfl_xor_sync(0xffffffff,add[r],1);
     add[r]+=__shfl_xor_sync(0xffffffff,add[r],2); sum[r]+=add[r];
   }
   __syncwarp();
   #pragma unroll
   for(int n=0;n<4;n++) {
     #pragma unroll
     for(int i=0;i<8;i++) acc[n].x[i]*=alpha[(i&2)/2];
     #pragma unroll
     for(int d=0;d<2;d++) {
       wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> p;
       wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> v;
       wmma::load_matrix_sync(p,P+warp*16*N+d*16,N);
       wmma::load_matrix_sync(v,V+d*16*D+n*16,D);
       wmma::mma_sync(acc[n],p,v,acc[n]);
     }
   }
   consumer_barrier();
 }
 long base=((long)(b*HKV+h)*SPLITS+split)*M;
 #pragma unroll
 for(int n=0;n<4;n++) wmma::store_matrix_sync(pa+(base+warp*16)*D+n*16,acc[n],D,wmma::mem_row_major);
 if(lane%4==0) for(int r=0;r<2;r++) {
   int row=warp*16+lane/4+r*8;
   pm[(base+row)*2]=mx[r]; pm[(base+row)*2+1]=sum[r];
 }
}
void attention(torch::Tensor qkv, torch::Tensor kc, torch::Tensor vc, torch::Tensor pt,
 torch::Tensor seq, torch::Tensor loc, torch::Tensor ks, torch::Tensor vs,
 torch::Tensor pa, torch::Tensor pm, int splits, int page, double scale, int stages) {
 int hkv=kc.size(2), hq=(qkv.size(2)/64-2*hkv)/2;
 auto stream=at::cuda::getCurrentCUDAStream();
 dim3 grid(splits,qkv.size(0),hkv);
 #define LAUNCH(S) verify<S><<<grid,160,0,stream>>>((__nv_bfloat16*)qkv.data_ptr(),(unsigned char*)kc.data_ptr(),(unsigned char*)vc.data_ptr(),pt.data_ptr<int>(),seq.data_ptr<int>(),loc.data_ptr<long>(),ks.data_ptr<float>(),vs.data_ptr<float>(),pa.data_ptr<float>(),pm.data_ptr<float>(),hq,hkv,pt.size(1),page,splits,scale)
 if(stages==2) { LAUNCH(2); } else if(stages==4) { LAUNCH(4); } else { LAUNCH(3); }
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''
module = load_inline(name='codex_verify_v1', cpp_sources=CPP, cuda_sources=CUDA, functions=['attention'],
                    extra_cuda_cflags=['-O3','--use_fast_math','-lineinfo','--ptxas-options=-v'],
                    extra_cflags=['-O3'], verbose=True)

def run(data, splits=None, stages=3):
    import submission, triton
    c=data['config']; b=c['batch']; h=c['num_kv_heads']; g=c['num_q_heads']//h
    assert c['q_len']==8 and g==8 and c['head_dim']==64 and not c['window']
    s=splits or max(1,528//(b*h))
    qkv=data['qkv']; device=qkv.device
    pa=torch.empty((b,h,s,64,64),dtype=torch.float32,device=device)
    pm=torch.empty((b,h,s,64,2),dtype=torch.float32,device=device)
    out=torch.empty((b,8,c['num_q_heads']*64),dtype=torch.bfloat16,device=device)
    module.attention(qkv,data['k_cache'],data['v_cache'],data['page_table'],data['seq_lens'],data['cache_loc'],
                     data['k_scale'],data['v_scale'],pa,pm,s,c['page_size'],c['sm_scale']*c['qk_scale_factor']*submission.LOG2E,stages)
    submission._combine[(b*8,c['num_q_heads'])](qkv,pa,pm,data['v_scale'],out,HQ=c['num_q_heads'],HKV=h,G=g,L=8,M=64,D=64,S=s,S_PAD=triton.next_power_of_2(s),SB=min(128,triton.next_power_of_2(s)),PDL=False,num_warps=4 if s>32 else 2)
    return out

"""Hopper paired-head native attention experiment; never dispatched by submission.py.

The best session-1 variant uses a producer warp with cp.async and four consumer warps,
FP8 WGMMA QK, FP16 register-to-shared MMA PV, and the existing Triton split reduction.
Only full attention, eight query tokens, group size eight, D=64, pages 1/64 is supported.
Build targets sm_90a (H100/H200), at import time. See NOTES-codex.md for measurements.
"""
import torch
from pathlib import Path
from torch.utils.cpp_extension import load_inline

CPP = r'''
#include <torch/extension.h>
void attention(torch::Tensor qkv, torch::Tensor kc, torch::Tensor vc, torch::Tensor pt,
 torch::Tensor seq, torch::Tensor loc, torch::Tensor ks, torch::Tensor vs,
 torch::Tensor pa, torch::Tensor pm, int splits, int page, double scale, int stages);
'''
CUDA = r'''
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
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
__device__ __forceinline__ void consumer_barrier(unsigned wg) {
 if(wg==0)asm volatile("bar.sync 1,128;":::"memory"); else asm volatile("bar.sync 2,128;":::"memory");
}
__device__ __forceinline__ unsigned acquire(unsigned* p) {
 unsigned v; asm volatile("ld.acquire.cta.shared.u32 %0, [%1];":"=r"(v):"r"((unsigned)__cvta_generic_to_shared(p)):"memory"); return v;
}
__device__ __forceinline__ void release(unsigned* p,unsigned v) {
 asm volatile("st.release.cta.shared.u32 [%0], %1;"::"r"((unsigned)__cvta_generic_to_shared(p)),"r"(v):"memory");
}

__device__ __forceinline__ void mbwait(unsigned long long* p,unsigned phase) {
 unsigned done;
 do {asm volatile("{.reg .pred p; mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p,[%1],%2; selp.b32 %0,1,0,p;}" : "=r"(done) : "r"((unsigned)__cvta_generic_to_shared(p)),"r"(phase):"memory");}while(!done);
}
__device__ __forceinline__ void mbarrive(unsigned long long* p) {
 asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _,[%0];"::"r"((unsigned)__cvta_generic_to_shared(p)):"memory");
}
template<int STAGES, int PAGE>
__global__ __launch_bounds__(288,2) void verify(const __nv_bfloat16* __restrict__ qkv, unsigned char* kc, unsigned char* vc,
 const int* __restrict__ pt, const int* __restrict__ seqs, const long* __restrict__ loc,
 const float* ks, const float* vs, float* __restrict__ pa, float* __restrict__ pm,
 int HQ,int HKV,int MP,int SPLITS,float SCALE) {
 constexpr int N=32, M=64, D=64;
 __shared__ __align__(1024) unsigned char raw[STAGES][2][N*128];
 __shared__ __align__(16) unsigned char own[2][2][8*D];
 __shared__ __align__(32) half V[2][N*(D+8)];
 __shared__ __align__(1024) unsigned char transposed_k[N*128];
 __shared__ __align__(16) unsigned char Q[2][M*(D+16)];
 __shared__ unsigned long long ready[STAGES],empty[STAGES];
 const unsigned global_tid=threadIdx.x, wg=global_tid/128, tid=global_tid%128, warp=tid/32, lane=tid%32;
 const int b=blockIdx.y, h=blockIdx.z*2+wg, split=blockIdx.x;
 const int seq=seqs[b], width=(2*HQ+2*HKV)*D, G=8;
 const int chunk=((seq+N*SPLITS-1)/(N*SPLITS))*N;
 const int lo=split*chunk, hi=min(seq,lo+chunk);
 const int tiles=max(0,(hi-lo+N-1)/N)+(split==0);
 const float kscale=*ks, vscale=*vs;
 if(global_tid<STAGES) {
  asm volatile("mbarrier.init.shared.b64 [%0],1;"::"r"((unsigned)__cvta_generic_to_shared(ready+global_tid)):"memory");
  asm volatile("mbarrier.init.shared.b64 [%0],2;"::"r"((unsigned)__cvta_generic_to_shared(empty+global_tid)):"memory");
 }
 // Each warp normalizes one row at a time, exactly the same query FP8 rounding as FA3.
 if(wg<2) for(int row=warp;row<M;row+=4) {
   int tok=b*8+row/G, head=h*G+row%G;
   float x=__bfloat162float(qkv[tok*width+head*D+lane]);
   float y=__bfloat162float(qkv[tok*width+head*D+lane+32]);
   float r=rsqrtf(warp_sum(x*x+y*y)/64.f+1e-5f);
   Q[wg][row*(D+16)+lane]=to8(x*r); Q[wg][row*(D+16)+lane+32]=to8(y*r);
 }
 // New keys are handled without depending on concurrent writes to the shared cache.
 if(wg<2 && split==0) for(int row=warp;row<8;row+=4) {
   int tok=b*8+row;
   float x=__bfloat162float(qkv[tok*width+(HQ+h)*D+lane]);
   float y=__bfloat162float(qkv[tok*width+(HQ+h)*D+lane+32]);
   float r=rsqrtf(warp_sum(x*x+y*y)/64.f+1e-5f)/kscale;
   unsigned char k0=to8(x*r), k1=to8(y*r);
   unsigned char v0=to8(__bfloat162float(qkv[tok*width+(HQ+HKV+h)*D+lane])/vscale);
   unsigned char v1=to8(__bfloat162float(qkv[tok*width+(HQ+HKV+h)*D+lane+32])/vscale);
   own[wg][0][row*D+lane]=k0; own[wg][0][row*D+lane+32]=k1;
   own[wg][1][row*D+lane]=v0; own[wg][1][row*D+lane+32]=v1;
   long slot=loc[tok]*HKV*D+h*D;
   kc[slot+lane]=k0; kc[slot+lane+32]=k1;
   vc[slot+lane]=v0; vc[slot+lane+32]=v1;
 }
 __syncthreads();
 if(wg>=2) {
   // Several distinct gather groups remain in flight while consumers run.
   for(int step=0;step<tiles;step++) {
       int stage=step%STAGES;
       if(step>=STAGES)mbwait(empty+stage,((step/STAGES)-1)&1);
       bool isnew=split==0 && step==tiles-1;
       if(!isnew) {
         if(lane==0)asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _,[%0],%1;"::"r"((unsigned)__cvta_generic_to_shared(ready+stage)),"r"(N*128*2):"memory");
         __syncwarp();
         int t=min(lo+step*N+lane,hi-1);
         int row=pt[b*MP+t/PAGE]*PAGE+t%PAGE;
         unsigned smk=__cvta_generic_to_shared(raw[stage][0]+lane*128),smv=__cvta_generic_to_shared(raw[stage][1]+lane*128),bar=__cvta_generic_to_shared(ready+stage);
         asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1],128,[%2];"::"r"(smk),"l"(kc+(long)row*HKV*D+blockIdx.z*128),"r"(bar):"memory");
         asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1],128,[%2];"::"r"(smv),"l"(vc+(long)row*HKV*D+blockIdx.z*128),"r"(bar):"memory");
       } else {
         for(int i=lane;i<N*128;i+=32) {
           unsigned rr=i/128,hh=(i%128)/64,dd=i%64;
           raw[stage][0][i]=rr<8?own[hh][0][rr*64+dd]:0;
           raw[stage][1][i]=rr<8?own[hh][1][rr*64+dd]:0;
         }
         __syncwarp();
         asm volatile("fence.proxy.async.shared::cta;":::"memory");
         if(lane==0)mbarrive(ready+stage);
       }
   }
   return;
 }
 wmma::fragment<wmma::accumulator,16,16,16,float> acc[4];
 #pragma unroll
 for(int n=0;n<4;n++) wmma::fill_fragment(acc[n],0.f);
 float mx[2]={-1e30f,-1e30f}, sum[2]={0,0};
 unsigned q[2][4];
 #pragma unroll
 for(int d=0;d<2;d++) {
   #pragma unroll
   for(int i=0;i<4;i++) {
     unsigned row=warp*16+lane/4+(i%2)*8, col=d*32+(lane%4)*4+(i/2)*16;
     q[d][i]=*(unsigned*)(Q[wg]+row*(D+16)+col);
   }
 }
 for(int tile=0;tile<tiles;tile++) {
   int stage=tile%STAGES;
   mbwait(ready+stage,(tile/STAGES)&1);
   unsigned ro=tid/4,dd=(tid%4)*16,off=ro*128+wg*64+dd;
   *(uint4*)(transposed_k+(off^((ro&7)<<4)))=*(const uint4*)(raw[stage][0]+off);
   asm volatile("fence.proxy.async.shared::cta;":::"memory");
   #pragma unroll
   for(unsigned j=0;j<N*D/256;j++) {
     unsigned i=2*tid+j*256;
     unsigned target=(i/D)*(D+8)+i%D;
     __half2_raw vv=__nv_cvt_fp8x2_to_halfraw2(*(unsigned short*)(raw[stage][1]+(i/D)*128+wg*64+i%D),__NV_E4M3);
     *(half2*)(V[wg]+target)=vv;
   }
   consumer_barrier(wg);
   wmma::fragment<wmma::accumulator,16,16,16,float> score[2];
   wmma::fill_fragment(score[0],0.f); wmma::fill_fragment(score[1],0.f);
   asm volatile("" : "+f"(score[0].x[0]), "+f"(score[0].x[1]), "+f"(score[0].x[2]), "+f"(score[0].x[3]), "+f"(score[0].x[4]), "+f"(score[0].x[5]), "+f"(score[0].x[6]), "+f"(score[0].x[7]), "+f"(score[1].x[0]), "+f"(score[1].x[1]), "+f"(score[1].x[2]), "+f"(score[1].x[3]), "+f"(score[1].x[4]), "+f"(score[1].x[5]), "+f"(score[1].x[6]), "+f"(score[1].x[7]) :: "memory");
   asm volatile("wgmma.fence.sync.aligned;":::"memory");
   #pragma unroll
   for(int d=0;d<2;d++) {
     unsigned long desc=(((unsigned long)__cvta_generic_to_shared(transposed_k+wg*64+d*32)&0x3ffff)>>4)
                        | (1ull<<16) | (64ull<<32) | (1ull<<62);
     asm volatile("wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, 1, 1, 1;"
       : "+f"(score[0].x[0]), "+f"(score[0].x[1]), "+f"(score[0].x[2]), "+f"(score[0].x[3]), "+f"(score[0].x[4]), "+f"(score[0].x[5]), "+f"(score[0].x[6]), "+f"(score[0].x[7]), "+f"(score[1].x[0]), "+f"(score[1].x[1]), "+f"(score[1].x[2]), "+f"(score[1].x[3]), "+f"(score[1].x[4]), "+f"(score[1].x[5]), "+f"(score[1].x[6]), "+f"(score[1].x[7])
       : "r"(q[d][0]),"r"(q[d][1]),"r"(q[d][2]),"r"(q[d][3]),"l"(desc));
   }
   asm volatile("wgmma.commit_group.sync.aligned;\nwgmma.wait_group.sync.aligned 0;":::"memory");
   asm volatile("" : "+f"(score[0].x[0]), "+f"(score[0].x[1]), "+f"(score[0].x[2]), "+f"(score[0].x[3]), "+f"(score[0].x[4]), "+f"(score[0].x[5]), "+f"(score[0].x[6]), "+f"(score[0].x[7]), "+f"(score[1].x[0]), "+f"(score[1].x[1]), "+f"(score[1].x[2]), "+f"(score[1].x[3]), "+f"(score[1].x[4]), "+f"(score[1].x[5]), "+f"(score[1].x[6]), "+f"(score[1].x[7]) :: "memory");
   bool isnew=split==0 && tile==tiles-1;
   // Mask only the final cached tile and the new-token tile.
   if(isnew || lo+(tile+1)*N>hi) {
     int bounds[2]={isnew?int(warp)*2:hi-lo-tile*N-1,isnew?int(warp)*2+1:hi-lo-tile*N-1};
     #pragma unroll
     for(int n=0;n<2;n++) {
       #pragma unroll
       for(int i=0;i<8;i++) {
         int r=(i&2)/2;
         int col=n*16+(lane%4)*2+(i&1)+(i&4)*2;
         if(col>bounds[r]) score[n].x[i]=-1e30f;
       }
     }
   }
   float now[2]={-1e30f,-1e30f};
   #pragma unroll
   for(int n=0;n<2;n++) {
     #pragma unroll
     for(int i=0;i<8;i++) now[(i&2)/2]=fmaxf(now[(i&2)/2],score[n].x[i]);
   }
   float alpha[2];
   #pragma unroll
   for(int r=0;r<2;r++) {
     now[r]=fmaxf(now[r],__shfl_xor_sync(0xffffffff,now[r],1));
     now[r]=fmaxf(now[r],__shfl_xor_sync(0xffffffff,now[r],2));
     now[r]=fmaxf(now[r]*SCALE*kscale,mx[r]); alpha[r]=exp2f(mx[r]-now[r]);
     sum[r]*=alpha[r]; mx[r]=now[r];
   }
   float add[2]={0,0};
   #pragma unroll
   for(int n=0;n<2;n++) {
     #pragma unroll
     for(int i=0;i<8;i++) {
       int r=(i&2)/2;
       float p=exp2f(fmaf(score[n].x[i],SCALE*kscale,-mx[r]));
       add[r]+=p; score[n].x[i]=p;
     }
   }
   #pragma unroll
   for(int r=0;r<2;r++) {
     add[r]+=__shfl_xor_sync(0xffffffff,add[r],1);
     add[r]+=__shfl_xor_sync(0xffffffff,add[r],2); sum[r]+=add[r];
   }
   unsigned pr[2][4];
   #pragma unroll
   for(int d=0;d<2;d++) {
     #pragma unroll
     for(int i=0;i<4;i++) {
       half2 packed=__floats2half2_rn(score[d].x[2*i],score[d].x[2*i+1]);
       pr[d][i]=*(unsigned*)&packed;
     }
   }
   if(__any_sync(0xffffffff, alpha[0]!=1.f || alpha[1]!=1.f)) {
     #pragma unroll
     for(int n=0;n<4;n++) {
       #pragma unroll
       for(int i=0;i<8;i++) acc[n].x[i]*=alpha[(i&2)/2];
     }
   }
   #pragma unroll
   for(int n=0;n<4;n++) {
     #pragma unroll
     for(int d=0;d<2;d++) {
       #pragma unroll
       for(int j=0;j<2;j++) {
         unsigned addr=__cvta_generic_to_shared(V[wg]+(d*16+lane%16)*(D+8)+n*16+j*8);
         unsigned v0,v1;
         asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];":"=r"(v0),"=r"(v1):"r"(addr));
         float* z=acc[n].x+j*4;
         asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
           : "+f"(z[0]),"+f"(z[1]),"+f"(z[2]),"+f"(z[3])
           : "r"(pr[d][0]),"r"(pr[d][1]),"r"(pr[d][2]),"r"(pr[d][3]),"r"(v0),"r"(v1));
       }
     }
   }
   consumer_barrier(wg);
   if(tid==0)mbarrive(empty+stage);
 }
 long base=((long)(b*HKV+h)*SPLITS+split)*M;
 #pragma unroll
 for(int n=0;n<4;n++) wmma::store_matrix_sync(pa+(base+warp*16)*D+n*16,acc[n],D,wmma::mem_row_major);
 if(lane%4==0) for(int r=0;r<2;r++) {
   int row=warp*16+lane/4+r*8;
   pm[(base+row)*2]=mx[r]; pm[(base+row)*2+1]=sum[r];
 }
}
void attention(at::Tensor qkv, at::Tensor kc, at::Tensor vc, at::Tensor pt,
 at::Tensor seq, at::Tensor loc, at::Tensor ks, at::Tensor vs,
 at::Tensor pa, at::Tensor pm, int splits, int page, double scale, int stages) {
 int hkv=kc.size(2), hq=(qkv.size(2)/64-2*hkv)/2;
 auto stream=c10::cuda::getCurrentCUDAStream();
 dim3 grid(splits,qkv.size(0),hkv/2);
 #define LAUNCH(S,P) verify<S,P><<<grid,288,0,stream>>>((__nv_bfloat16*)qkv.data_ptr(),(unsigned char*)kc.data_ptr(),(unsigned char*)vc.data_ptr(),pt.data_ptr<int>(),seq.data_ptr<int>(),loc.data_ptr<long>(),ks.data_ptr<float>(),vs.data_ptr<float>(),pa.data_ptr<float>(),pm.data_ptr<float>(),hq,hkv,pt.size(1),splits,scale)
 if(page==1) {
   if(stages==2) { LAUNCH(2,1); } else if(stages==4) { LAUNCH(4,1); } else if(stages==5) { LAUNCH(5,1); } else if(stages==6) { LAUNCH(3,1); } else if(stages==7) { LAUNCH(3,1); } else { LAUNCH(3,1); }
 } else {
   if(stages==2) { LAUNCH(2,64); } else if(stages==4) { LAUNCH(4,64); } else if(stages==5) { LAUNCH(5,64); } else if(stages==6) { LAUNCH(3,64); } else if(stages==7) { LAUNCH(3,64); } else { LAUNCH(3,64); }
 }
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''
_build = Path(__file__).resolve().parent / '.build-pair-bulk'
_build.mkdir(exist_ok=True)
module = load_inline(name='codex_verify_pair3bulk', build_directory=str(_build), cpp_sources=CPP, cuda_sources=CUDA, functions=['attention'],
                    extra_cuda_cflags=['-gencode=arch=compute_90a,code=sm_90a','-O3','--use_fast_math','-lineinfo','--ptxas-options=-v'],
                    extra_cflags=['-O3'], extra_ldflags=['-lcuda'], verbose=True)

def run(data, splits=None, stages=3):
    import submission, triton
    c=data['config']; b=c['batch']; h=c['num_kv_heads']; g=c['num_q_heads']//h
    assert h%2==0 and c['q_len']==8 and g==8 and c['head_dim']==64 and not c['window'] and c['page_size'] in (1,64)
    s=splits or max(1,264//(b*h//2))
    qkv=data['qkv']; device=qkv.device
    pa=torch.empty((b,h,s,64,64),dtype=torch.float32,device=device)
    pm=torch.empty((b,h,s,64,2),dtype=torch.float32,device=device)
    out=torch.empty((b,8,c['num_q_heads']*64),dtype=torch.bfloat16,device=device)
    module.attention(qkv,data['k_cache'],data['v_cache'],data['page_table'],data['seq_lens'],data['cache_loc'],
                     data['k_scale'],data['v_scale'],pa,pm,s,c['page_size'],c['sm_scale']*c['qk_scale_factor']*submission.LOG2E,stages)
    submission._combine[(b*8,c['num_q_heads'])](qkv,pa,pm,data['v_scale'],out,HQ=c['num_q_heads'],HKV=h,G=g,L=8,M=64,D=64,S=s,S_PAD=triton.next_power_of_2(s),SB=min(128,triton.next_power_of_2(s)),PDL=False,num_warps=4 if s>32 else 2)
    return out

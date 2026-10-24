# okc: problem=muse_attention_decode
"""Gated attention decode with Triton attention and a a native CUDA full-layer merge for small batches.

1. _attn_split: one program per (split of the context, request, KV head). It normalizes (and
   on sliding layers ropes) its group's queries straight from the fused projection output,
   streams its slice of the paged FP8 cache through an online softmax, and writes a partial
   (max, sum, accumulator). The split-0 program also normalizes/ropes the block's new keys,
   quantizes the new keys and values to FP8, stores them in the cache, and attends them
   causally from registers.
2. Merge kernels reduce the splits per query head, then apply RMSNorm and sigmoid(gate).
   Small full-layer batches use an eight-warp native CUDA reduction."""

from typing import TypedDict

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

try:  # Gluon (Triton's low-level dialect, ships with triton 3.8): the verify full-layer kernel.
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language.nvidia.hopper import (fence_async_shared, mbarrier, warpgroup_mma,
                                                                    warpgroup_mma_wait)
    from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp
    HAVE_GLUON = True
except ImportError:
    HAVE_GLUON = False


class Config(TypedDict):
    batch: int            # B
    q_len: int            # L: 1 for decode, 8 for a speculative verify block (causal inside it)
    num_q_heads: int      # Hq: 16 per GPU under TP8
    num_kv_heads: int     # Hkv: 2; each serves Hq / Hkv query heads
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers; 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)


class Inputs(TypedDict):
    qkv: torch.Tensor          # bfloat16 [B, L, (2*Hq + 2*Hkv)*64], 2304 under tp8: per token q (Hq*64),
                               # then the block's keys before norm and RoPE (Hkv*64), values
                               # (Hkv*64) and the output gate's logits (Hq*64)
    positions: torch.Tensor    # int64 [B*L], each new token's position (seq_lens[b] + i)
    cache_loc: torch.Tensor    # int64 [B*L], each new token's flat cache row (page * page_size + offset)
    cache_seqlens: torch.Tensor  # int32 [B], seq_lens + L: what each request attends over
    cu_seqlens_q: torch.Tensor   # int32 [B+1], 0, L, 2L, ...: the block's query offsets
    scheduler_metadata: torch.Tensor | None  # int32, FA3's tile schedule (get_scheduler_metadata) on
                                             # full-attention decode steps, else None
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64], written in place
    k_scale: torch.Tensor     # float32 [], key = e4m3 * k_scale
    v_scale: torch.Tensor     # float32 [], value = e4m3 * v_scale
    page_table: torch.Tensor  # int32 [B, max_pages], physical page of each logical page (-1 unused)
    seq_lens: torch.Tensor    # int32 [B], cached tokens per request; the block starts at this position
    config: Config


# Eight native warps merge each head's FP32 full-layer partials, then normalize and gate.
# Build once at import; input-derived data is read only inside the timed kernels.
from torch.utils.cpp_extension import load_inline

_NATIVE_MERGE = load_inline(
    name="muse_native_merge",
    cpp_sources=r'''

#include <torch/extension.h>
void combine_full(torch::Tensor qkv,torch::Tensor pa,torch::Tensor pm,torch::Tensor vs,torch::Tensor out,int hq,int hkv,int length,int splits,bool pdl);
''',
    cuda_sources=r'''

#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
// Keep coherent partial loads after griddepcontrol.wait. Ordinary const loads
// can be hoisted across that dependency wait by nvcc.
__device__ __forceinline__ float2 read2(const float* p) {
 float2 v;
 asm volatile("ld.global.cg.v2.f32 {%0,%1}, [%2];" : "=f"(v.x), "=f"(v.y) : "l"(p) : "memory");
 return v;
}
__device__ __forceinline__ float sum32(float x) {
 #pragma unroll
 for(int d=16;d;d/=2)x+=__shfl_xor_sync(0xffffffff,x,d);
 return x;
}
__device__ __forceinline__ float max32(float x) {
 #pragma unroll
 for(int d=16;d;d/=2)x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,d));
 return x;
}
template<int W,int N,bool PDL>
__global__ __launch_bounds__(W*32) void reduce_kernel(const __nv_bfloat16* qkv,
 const float* pa,const float2* pm,const float* vs,__nv_bfloat16* out,
 int HQ,int HKV,int L,int B,int S) {
 unsigned lane=threadIdx.x%32,warp=threadIdx.x/32;
 int row=blockIdx.x;
 int tok=row/HQ,head=row%HQ,b=tok/L,g=HQ/HKV,h=head/g;
 int width=(2*HQ+2*HKV)*64,rr=(tok%L)*g+head%g,MG=L*g;
 int base=((b*HKV+h)*S)*MG+rr;
 float2 gate=__bfloat1622float2(*(const __nv_bfloat162*)(qkv+tok*width+(HQ+2*HKV)*64+head*64+2*lane));
 float vscale=*vs;
 if(PDL)asm volatile("griddepcontrol.wait;":::"memory");
 constexpr int NS=(N+31)/32;
 float2 ml[NS];
 float m=-1e30f;
 #pragma unroll
 for(int i=0;i<NS;i++) {
   int s=warp*N+i*32+lane;
   ml[i]=s<S && i*32+lane<N?read2((const float*)(pm+base+s*MG)):make_float2(-1e30f,0.f);
   m=fmaxf(m,ml[i].x);
 }
 m=max32(m);
 float weight[NS],den=0;
 #pragma unroll
 for(int i=0;i<NS;i++) {
   weight[i]=exp2f(ml[i].x-m);
   den=fmaf(weight[i],ml[i].y,den);
 }
 den=sum32(den);
 float2 a[N];
 #pragma unroll
 for(int i=0;i<N;i++) {
   int s=warp*N+i;
   a[i]=s<S?read2(pa+(base+s*MG)*64+2*lane):make_float2(0.f,0.f);
 }
 float x=0,y=0;
 #pragma unroll
 for(int i=0;i<N;i++) {
   float w=__shfl_sync(0xffffffff,weight[i/32],i%32);
   x=fmaf(w,a[i].x,x);y=fmaf(w,a[i].y,y);
 }
 __shared__ float partial[W][68];
 if(W>1) {
   partial[warp][2*lane]=x;partial[warp][2*lane+1]=y;
   if(lane==0){partial[warp][64]=m;partial[warp][65]=den;}
   __syncthreads();
   if(warp!=0)return;
   float gm=-1e30f;
   #pragma unroll
   for(int w=0;w<W;w++)gm=fmaxf(gm,partial[w][64]);
   x=0;y=0;den=0;
   #pragma unroll
   for(int w=0;w<W;w++) {
     float sc=exp2f(partial[w][64]-gm);
     x=fmaf(sc,partial[w][2*lane],x);y=fmaf(sc,partial[w][2*lane+1],y);
     den=fmaf(sc,partial[w][65],den);
   }
 }
 x=x/den*vscale;y=y/den*vscale;
 float sq=sum32(x*x+y*y),r=rsqrtf(sq/64.f+1e-5f);
 x=x*r/(1.f+expf(-gate.x));y=y*r/(1.f+expf(-gate.y));
 *(__nv_bfloat162*)(out+row*64+2*lane)=__floats2bfloat162_rn(x,y);
}

void combine_full(at::Tensor qkv,at::Tensor pa,at::Tensor pm,at::Tensor vs,at::Tensor out,int hq,int hkv,int length,int splits,bool pdl) {
 auto stream=c10::cuda::getCurrentCUDAStream();int b=qkv.size(0);
 cudaLaunchConfig_t config={};config.gridDim=dim3(b*length*hq);config.blockDim=dim3(256);config.stream=stream;
 cudaLaunchAttribute attr={};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;attr.val.programmaticStreamSerializationAllowed=pdl;
 config.attrs=&attr;config.numAttrs=1;
 #define FULL_RUN(P) C10_CUDA_CHECK(cudaLaunchKernelEx(&config,reduce_kernel<8,16,P>,(const __nv_bfloat16*)qkv.data_ptr(),pa.data_ptr<float>(),(const float2*)pm.data_ptr<float>(),vs.data_ptr<float>(),(__nv_bfloat16*)out.data_ptr(),hq,hkv,length,b,splits))
 if(pdl){FULL_RUN(true);}else{FULL_RUN(false);}
 #undef FULL_RUN
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}

''',
    functions=["combine_full"],
    extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3", "--use_fast_math"],
    extra_cflags=["-O3"],
    verbose=False,
)


RMS_EPS = 1e-5
LOG2E = 1.4426950408889634
ROPE_POSITIONS = 1 << 21

# Small-batch sliding decode: one 64-key tile per block, first gather before
# norm/RoPE, FP8 queries in registers, and FP32 split partials.
_NATIVE_SLIDING = load_inline(
    name="muse_native_query_cooperative_distributed_s33",
    cpp_sources=r'''#include <torch/extension.h>
void attention(torch::Tensor qkv,torch::Tensor cs,torch::Tensor kc,torch::Tensor vc,torch::Tensor pt,torch::Tensor seq,torch::Tensor loc,torch::Tensor ks,torch::Tensor vs,torch::Tensor pa,torch::Tensor pm,int page,double scale,bool early);
''',
    cuda_sources=r'''#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
__device__ __forceinline__ float sum32(float v) {
 #pragma unroll
 for(int d=16;d;d/=2)v+=__shfl_xor_sync(0xffffffff,v,d);
 return v;
}
__device__ __forceinline__ unsigned char to8(float x) {return __nv_cvt_float_to_fp8(x,__NV_SATFINITE,__NV_E4M3);}
__device__ __forceinline__ unsigned sw64(unsigned i) {return i ^ (((i>>7)&3)<<4);}
__device__ __forceinline__ void cp16(void* d,const void* s,int valid) {
 asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"((unsigned)__cvta_generic_to_shared(d)),"l"(s),"r"(valid?16:0):"memory");
}
// EARLY only changes when cached-key copies are issued, never the arithmetic.
template<int PAGE,bool EARLY>
__global__ __launch_bounds__(128) void sliding(const __nv_bfloat16* qkv,const float* cs,
 unsigned char* kc,unsigned char* vc,const int* pt,const int* seqs,const long* loc,
 const float* ks,const float* vs,float* pa,float* pm,int HQ,int HKV,int MP,int S,float scale) {
 __shared__ __align__(512) unsigned char K[4096];
 __shared__ __align__(16) unsigned char Vraw[4096];
 __shared__ __align__(1024) half V[4096];
 const int tid=threadIdx.x,warp=tid/32,lane=tid%32;
 const int b=blockIdx.y,h=blockIdx.z,split=blockIdx.x;
 const int seq=seqs[b],base=max(0,seq-2047),lo=base+split*64;
 const int width=(2*HQ+2*HKV)*64;
 asm volatile("griddepcontrol.launch_dependents;":::"memory");
 // Each thread starts two 16B copies per cache, over all 64 positions.
 auto gather=[&]() {
  #pragma unroll
  for(int j=0;j<2;j++) {
   int i=tid+j*128,row=i/4,piece=(i%4)*16,t=lo+row;
   bool valid=t<seq;
   int page=valid?pt[b*MP+t/PAGE]:0;
   long addr=((long)page*PAGE+t%PAGE)*HKV*64+h*64+piece;
   cp16(K+sw64(row*64+piece),kc+(valid?addr:0),valid);
   cp16(Vraw+row*64+piece,vc+(valid?addr:0),valid);
  }
  asm volatile("cp.async.commit_group;":::"memory");
 };
 if(EARLY)gather();
 float kscale=*ks,vscale=*vs;
 float c0=cs[(long)seq*64+lane/2],s0=cs[(long)seq*64+lane/2+32];
 float c1=cs[(long)seq*64+(lane+32)/2],s1=cs[(long)seq*64+(lane+32)/2+32];

 // Two real query heads per warp, sixteen lanes per head.
 const int qhead=warp*2+lane/16,qcol=(lane%16)*4;
 float x[4],sq=0;
 #pragma unroll
 for(int k=0;k<4;k++) {
  x[k]=__bfloat162float(qkv[b*width+(h*8+qhead)*64+qcol+k]);
  sq+=x[k]*x[k];
 }
 #pragma unroll
 for(int d=8;d;d/=2)sq+=__shfl_xor_sync(0xffffffff,sq,d,16);
 float qinv=rsqrtf(sq/64.f+1e-5f);
 unsigned packed=0;
 #pragma unroll
 for(int k=0;k<4;k+=2) {
  float cc=cs[(long)seq*64+(qcol+k)/2],ss=cs[(long)seq*64+(qcol+k)/2+32];
  float a=(x[k]*cc-x[k+1]*ss)*qinv,bb=(x[k+1]*cc+x[k]*ss)*qinv;
  packed|=(unsigned)to8(a)<<(8*k);packed|=(unsigned)to8(bb)<<(8*(k+1));
 }
 unsigned qr[2][4]={{0}};
 #pragma unroll
 for(int d=0;d<2;d++) {
  #pragma unroll
  for(int i=0;i<2;i++) {
   int src=((lane/4)*16+d*8+(lane%4)+i*4)&31;
   unsigned word=__shfl_sync(0xffffffff,packed,src);
   qr[d][i*2]=lane<8?word:0;
  }
 }
 // Keep new K/V in registers until the zero-filled cache gather completes.
 float nx=0,ny=0,nv0=0,nv1=0;
 bool own=seq>=lo && seq<lo+64;
 if(own && warp==1) {
  const __nv_bfloat16* p=qkv+b*width+(HQ+h)*64;
  nx=__bfloat162float(p[lane]);ny=__bfloat162float(p[lane+32]);
  float inv=rsqrtf(sum32(nx*nx+ny*ny)/64.f+1e-5f)/kscale;
  float xp=__shfl_xor_sync(0xffffffff,nx,1),yp=__shfl_xor_sync(0xffffffff,ny,1),sign=lane%2?1.f:-1.f;
  nx=(nx*c0+sign*xp*s0)*inv;ny=(ny*c1+sign*yp*s1)*inv;
  nv0=__bfloat162float(p[HKV*64+lane])/vscale;nv1=__bfloat162float(p[HKV*64+lane+32])/vscale;
 }
 if(!EARLY)gather();
 asm volatile("cp.async.wait_group 0;":::"memory");
 __syncthreads();
 if(own && warp==1) {
  int row=seq-lo;unsigned char k0=to8(nx),k1=to8(ny),v0=to8(nv0),v1=to8(nv1);
  K[sw64(row*64+lane)]=k0;K[sw64(row*64+lane+32)]=k1;
  Vraw[row*64+lane]=v0;Vraw[row*64+lane+32]=v1;
  long slot=loc[b]*HKV*64+h*64;
  kc[slot+lane]=k0;kc[slot+lane+32]=k1;vc[slot+lane]=v0;vc[slot+lane+32]=v1;
 }
 __syncthreads();
 #pragma unroll
 for(int j=0;j<4;j++) {
  unsigned i=8*tid+j*1024,target=i ^ (((i>>6)&7)<<3);
  unsigned long long bytes=*(unsigned long long*)(Vraw+i);unsigned cv[4];
  #pragma unroll
  for(int z=0;z<4;z++) {__half2_raw v=__nv_cvt_fp8x2_to_halfraw2((unsigned short)(bytes>>(16*z)),__NV_E4M3);cv[z]=*(unsigned*)&v;}
  asm volatile("st.shared.v4.u32 [%4],{%0,%1,%2,%3};"::"r"(cv[0]),"r"(cv[1]),"r"(cv[2]),"r"(cv[3]),"r"((unsigned)__cvta_generic_to_shared(V+target)):"memory");
 }
 asm volatile("fence.proxy.async.shared::cta;":::"memory");
 __syncthreads();
 float score[32]={0};
 asm volatile("" : "+f"(score[0]),"+f"(score[1]),"+f"(score[2]),"+f"(score[3]),"+f"(score[4]),"+f"(score[5]),"+f"(score[6]),"+f"(score[7]),"+f"(score[8]),"+f"(score[9]),"+f"(score[10]),"+f"(score[11]),"+f"(score[12]),"+f"(score[13]),"+f"(score[14]),"+f"(score[15]),"+f"(score[16]),"+f"(score[17]),"+f"(score[18]),"+f"(score[19]),"+f"(score[20]),"+f"(score[21]),"+f"(score[22]),"+f"(score[23]),"+f"(score[24]),"+f"(score[25]),"+f"(score[26]),"+f"(score[27]),"+f"(score[28]),"+f"(score[29]),"+f"(score[30]),"+f"(score[31]) :: "memory");
 asm volatile("wgmma.fence.sync.aligned;":::"memory");
 #pragma unroll
 for(int d=0;d<2;d++) {
  unsigned long desc=(((unsigned long)__cvta_generic_to_shared(K+d*32)&0x3ffff)>>4)|(1ull<<16)|(32ull<<32)|(2ull<<62);
  asm volatile("wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, 1, 1, 1;" : "+f"(score[0]),"+f"(score[1]),"+f"(score[2]),"+f"(score[3]),"+f"(score[4]),"+f"(score[5]),"+f"(score[6]),"+f"(score[7]),"+f"(score[8]),"+f"(score[9]),"+f"(score[10]),"+f"(score[11]),"+f"(score[12]),"+f"(score[13]),"+f"(score[14]),"+f"(score[15]),"+f"(score[16]),"+f"(score[17]),"+f"(score[18]),"+f"(score[19]),"+f"(score[20]),"+f"(score[21]),"+f"(score[22]),"+f"(score[23]),"+f"(score[24]),"+f"(score[25]),"+f"(score[26]),"+f"(score[27]),"+f"(score[28]),"+f"(score[29]),"+f"(score[30]),"+f"(score[31]) : "r"(qr[d][0]),"r"(qr[d][1]),"r"(qr[d][2]),"r"(qr[d][3]),"l"(desc));
 }
 asm volatile("wgmma.commit_group.sync.aligned;\nwgmma.wait_group.sync.aligned 0;":::"memory");
 asm volatile("" : "+f"(score[0]),"+f"(score[1]),"+f"(score[2]),"+f"(score[3]),"+f"(score[4]),"+f"(score[5]),"+f"(score[6]),"+f"(score[7]),"+f"(score[8]),"+f"(score[9]),"+f"(score[10]),"+f"(score[11]),"+f"(score[12]),"+f"(score[13]),"+f"(score[14]),"+f"(score[15]),"+f"(score[16]),"+f"(score[17]),"+f"(score[18]),"+f"(score[19]),"+f"(score[20]),"+f"(score[21]),"+f"(score[22]),"+f"(score[23]),"+f"(score[24]),"+f"(score[25]),"+f"(score[26]),"+f"(score[27]),"+f"(score[28]),"+f"(score[29]),"+f"(score[30]),"+f"(score[31]) :: "memory");
 float mx[2]={-1e30f,-1e30f},sum[2]={0,0};
 #pragma unroll
 for(int i=0;i<32;i++) {
  int col=(i/8)*16+(lane%4)*2+(i&1)+(i&4)*2;
  score[i]=lo+col<=seq?score[i]:-1e30f;
  mx[(i&2)/2]=fmaxf(mx[(i&2)/2],score[i]);
 }
 #pragma unroll
 for(int r=0;r<2;r++) {
  mx[r]=fmaxf(mx[r],__shfl_xor_sync(0xffffffff,mx[r],1));mx[r]=fmaxf(mx[r],__shfl_xor_sync(0xffffffff,mx[r],2));
  mx[r]=max(-1e30f,mx[r]*scale*kscale);
 }
 #pragma unroll
 for(int i=0;i<32;i++) {
  int col=(i/8)*16+(lane%4)*2+(i&1)+(i&4)*2,r=(i&2)/2;
  float p=lo+col<=seq?exp2f(fmaf(score[i],scale*kscale,-mx[r])):0.f;
  score[i]=p;sum[r]+=p;
 }
 #pragma unroll
 for(int r=0;r<2;r++) {sum[r]+=__shfl_xor_sync(0xffffffff,sum[r],1);sum[r]+=__shfl_xor_sync(0xffffffff,sum[r],2);}
 unsigned pr[4][4];
 #pragma unroll
 for(int d=0;d<4;d++) {
  #pragma unroll
  for(int i=0;i<4;i++) {half2 p=__floats2half2_rn(score[d*8+2*i],score[d*8+2*i+1]);pr[d][i]=*(unsigned*)&p;}
 }
 float acc[32]={0};
 asm volatile("" : "+f"(acc[0]),"+f"(acc[1]),"+f"(acc[2]),"+f"(acc[3]),"+f"(acc[4]),"+f"(acc[5]),"+f"(acc[6]),"+f"(acc[7]),"+f"(acc[8]),"+f"(acc[9]),"+f"(acc[10]),"+f"(acc[11]),"+f"(acc[12]),"+f"(acc[13]),"+f"(acc[14]),"+f"(acc[15]),"+f"(acc[16]),"+f"(acc[17]),"+f"(acc[18]),"+f"(acc[19]),"+f"(acc[20]),"+f"(acc[21]),"+f"(acc[22]),"+f"(acc[23]),"+f"(acc[24]),"+f"(acc[25]),"+f"(acc[26]),"+f"(acc[27]),"+f"(acc[28]),"+f"(acc[29]),"+f"(acc[30]),"+f"(acc[31]) :: "memory");
 asm volatile("wgmma.fence.sync.aligned;":::"memory");
 #pragma unroll
 for(int d=0;d<4;d++) {
  unsigned long desc=(((unsigned long)__cvta_generic_to_shared(V+d*1024)&0x3ffff)>>4)|(64ull<<16)|(64ull<<32)|(1ull<<62);
  asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, 1, 1, 1, 1;" : "+f"(acc[0]),"+f"(acc[1]),"+f"(acc[2]),"+f"(acc[3]),"+f"(acc[4]),"+f"(acc[5]),"+f"(acc[6]),"+f"(acc[7]),"+f"(acc[8]),"+f"(acc[9]),"+f"(acc[10]),"+f"(acc[11]),"+f"(acc[12]),"+f"(acc[13]),"+f"(acc[14]),"+f"(acc[15]),"+f"(acc[16]),"+f"(acc[17]),"+f"(acc[18]),"+f"(acc[19]),"+f"(acc[20]),"+f"(acc[21]),"+f"(acc[22]),"+f"(acc[23]),"+f"(acc[24]),"+f"(acc[25]),"+f"(acc[26]),"+f"(acc[27]),"+f"(acc[28]),"+f"(acc[29]),"+f"(acc[30]),"+f"(acc[31]) : "r"(pr[d][0]),"r"(pr[d][1]),"r"(pr[d][2]),"r"(pr[d][3]),"l"(desc));
 }
 asm volatile("wgmma.commit_group.sync.aligned;\nwgmma.wait_group.sync.aligned 0;":::"memory");
 asm volatile("" : "+f"(acc[0]),"+f"(acc[1]),"+f"(acc[2]),"+f"(acc[3]),"+f"(acc[4]),"+f"(acc[5]),"+f"(acc[6]),"+f"(acc[7]),"+f"(acc[8]),"+f"(acc[9]),"+f"(acc[10]),"+f"(acc[11]),"+f"(acc[12]),"+f"(acc[13]),"+f"(acc[14]),"+f"(acc[15]),"+f"(acc[16]),"+f"(acc[17]),"+f"(acc[18]),"+f"(acc[19]),"+f"(acc[20]),"+f"(acc[21]),"+f"(acc[22]),"+f"(acc[23]),"+f"(acc[24]),"+f"(acc[25]),"+f"(acc[26]),"+f"(acc[27]),"+f"(acc[28]),"+f"(acc[29]),"+f"(acc[30]),"+f"(acc[31]) :: "memory");
 if(lane<8) {
  long outbase=((long)(b*HKV+h)*S+split)*8+warp*2;
  #pragma unroll
  for(int i=0;i<32;i++)if(!(i&2)) {
   int row=lane/4,col=(i/8)*16+(lane%4)*2+(i&1)+(i&4)*2;
   pa[(outbase+row)*64+col]=acc[i];
  }
  if(lane%4==0){pm[(outbase+lane/4)*2]=mx[0];pm[(outbase+lane/4)*2+1]=sum[0];}
 }
}
void attention(at::Tensor qkv,at::Tensor cs,at::Tensor kc,at::Tensor vc,at::Tensor pt,at::Tensor seq,
 at::Tensor loc,at::Tensor ks,at::Tensor vs,at::Tensor pa,at::Tensor pm,int page,double scale,bool early) {
 int hkv=kc.size(2),hq=(qkv.size(2)/64-2*hkv)/2,S=pa.size(2);
 dim3 grid(S,qkv.size(0),hkv);auto stream=c10::cuda::getCurrentCUDAStream();
 #define RUN(P,E) sliding<P,E><<<grid,128,0,stream>>>((const __nv_bfloat16*)qkv.data_ptr(),cs.data_ptr<float>(),(unsigned char*)kc.data_ptr(),(unsigned char*)vc.data_ptr(),pt.data_ptr<int>(),seq.data_ptr<int>(),loc.data_ptr<long>(),ks.data_ptr<float>(),vs.data_ptr<float>(),pa.data_ptr<float>(),pm.data_ptr<float>(),hq,hkv,pt.size(1),S,scale)
 if(page==1){if(early){RUN(1,true);}else{RUN(1,false);}}else{if(early){RUN(64,true);}else{RUN(64,false);}}
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
''',
    functions=["attention"],
    extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3", "--use_fast_math"],
    extra_cflags=["-O3"],
    verbose=False,
)

_ROPE = {}
_SMS = {}
_COUNTERS = {}
# The MMA input precision for P@V (and QK^T without qk8). FP16 holds every E4M3 value exactly and
# converts from it in one instruction (BF16 takes two); P in [0, 1] keeps more bits than in BF16.
# A test hook too: Triton's CPU interpreter can't multiply 16-bit floats.
DOT_DTYPE = tl.float16
# Programmatic dependent launch of the merge kernel (off in the CPU interpreter).
USE_PDL = True
print(f"env: torch {torch.__version__}, triton {triton.__version__}", flush=True)


def _cos_sin(theta, d, device):
    """[2^21, d] float32 cos | sin, built as SGLang's RotaryEmbedding (and the reference) do."""
    cache = _ROPE.get(theta)
    if cache is None:
        inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d))
        ang = torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device)[:, None] * inv_freq
        cache = _ROPE[theta] = torch.cat((ang.cos(), ang.sin()), dim=-1).contiguous()
    return cache


def _counters(n, device):
    """Zeroed int32 arrival counters for the fused merge; each call leaves them zeroed again."""
    buf = _COUNTERS.get(device)
    if buf is None or buf.numel() < n:
        buf = _COUNTERS[device] = torch.zeros(max(n, 4096), dtype=torch.int32, device=device)
    return buf


def _num_sms(device):
    if device.type != "cuda":
        return 132
    if device.index not in _SMS:
        _SMS[device.index] = torch.cuda.get_device_properties(device).multi_processor_count
    return _SMS[device.index]


# ---- Verify blocks on full layers (Gluon, warp-specialized) ----
# One program per (split, request, KV head pair), FA3 style: a 4-warp producer gathers the pair's
# 128-byte K and V rows (page indices, cp.async) into an NS-slot shared ring and signals `ready`
# (cp.async.mbarrier.arrive); one consumer warpgroup per KV head runs QK^T (FP8 wgmma on its 64-byte
# half of each row), stages its V half as FP16, frees the slot (`empty`), and does the online softmax
# and P@V. The extra split S holds the block's new tokens (normalized, stored as FP8, attended causally).
if HAVE_GLUON:
    @gluon.jit
    def _gw_producer(pt_row, kbase, vbase, k_ring, v_ring, ready, empty, lo, hi, ntiles,
                     PAGE: gl.constexpr, ROW_W: gl.constexpr, BN: gl.constexpr, C: gl.constexpr,
                     NS: gl.constexpr, PW: gl.constexpr):
        GLY: gl.constexpr = gl.BlockedLayout([1, 16], [4, 8], [PW, 1], [1, 0])
        n_g = gl.arange(0, BN, layout=gl.SliceLayout(1, GLY))
        c_g = gl.arange(0, C, layout=gl.SliceLayout(0, GLY))
        t = gl.maximum(gl.minimum(lo + n_g, hi - 1), 0)
        pg = gl.load(pt_row + t // PAGE)
        for i in range(ntiles):
            s = i % NS
            off = ((pg.to(gl.int64) * PAGE + t % PAGE) * ROW_W)[:, None] + c_g[None, :]
            # Next tile's page indices before waiting for the slot.
            t = gl.maximum(gl.minimum(lo + (i + 1) * BN + n_g, hi - 1), 0)
            pg = gl.load(pt_row + t // PAGE)
            mbarrier.wait(empty.index(s), ((i // NS) & 1) ^ 1)
            cp.async_copy_global_to_shared(k_ring.index(s), kbase + off)
            cp.async_copy_global_to_shared(v_ring.index(s), vbase + off)
            cp.mbarrier_arrive(ready.index(s), increment_count=False)

    @gluon.jit
    def _gw_store(pacc_ptr, pml_ptr, b, hh, split, m_i, l_i, acc, HKV: gl.constexpr, SP: gl.constexpr,
                  M: gl.constexpr, D: gl.constexpr, R_L: gl.constexpr, A_L: gl.constexpr):
        ro = gl.arange(0, M, layout=gl.SliceLayout(1, A_L))
        co = gl.arange(0, D, layout=gl.SliceLayout(0, A_L))
        part = ((b * HKV + hh) * SP + split) * M + ro
        gl.store(pacc_ptr + part[:, None] * D + co[None, :], acc)
        rs = gl.arange(0, M, layout=R_L)
        parts = ((b * HKV + hh) * SP + split) * M + rs
        gl.store(pml_ptr + parts * 2, m_i)
        gl.store(pml_ptr + parts * 2 + 1, l_i)

    @gluon.jit
    def _gw_consumer(q, k_ring, v_ring, vh, ready, empty, lo, hi, ntiles, scale, pacc_ptr, pml_ptr, b, hh,
                     split, qkv_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, loc_ptr,
                     HEAD: gl.constexpr, HQ: gl.constexpr, HKV: gl.constexpr, G: gl.constexpr, L: gl.constexpr,
                     S: gl.constexpr, M: gl.constexpr, D: gl.constexpr, BN: gl.constexpr, NS: gl.constexpr):
        VL: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
        S_L: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1],
                                                      instr_shape=[16, BN, 32])
        O_L: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1],
                                                      instr_shape=[16, D, 16])
        P_L: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=O_L, k_width=2)
        m_i = gl.full([M], -1.0e30, gl.float32, layout=gl.SliceLayout(1, S_L))
        l_i = gl.zeros([M], gl.float32, layout=gl.SliceLayout(1, S_L))
        acc = gl.zeros([M, D], gl.float32, layout=O_L)
        n_s = gl.arange(0, BN, layout=gl.SliceLayout(0, S_L))
        for i in range(ntiles):
            s = i % NS
            mbarrier.wait(ready.index(s), (i // NS) & 1)
            s_tok = warpgroup_mma(q, k_ring.index(s).slice(HEAD * D, D, 1).permute((1, 0)),
                                  gl.zeros([M, BN], gl.float32, S_L), use_acc=False, is_async=True)
            vh.store(v_ring.index(s).slice(HEAD * D, D, 1).load(VL).to(gl.float16))
            fence_async_shared()
            t = lo + i * BN + n_s
            sc = warpgroup_mma_wait(0, deps=[s_tok])
            gl.barrier()
            mbarrier.arrive(empty.index(s))
            sc = gl.where((t < hi)[None, :], sc, float("-inf"))
            m_new = gl.maximum(m_i, gl.max(sc, 1) * scale)
            alpha = gl.exp2(m_i - m_new)
            p = gl.exp2(gl.fma(sc, scale, -m_new[:, None]))
            l_i = l_i * alpha + gl.sum(p, 1)
            m_i = m_new
            acc = acc * gl.convert_layout(alpha, gl.SliceLayout(1, O_L))[:, None]
            p16 = gl.convert_layout(p.to(gl.float16), P_L)
            a_tok = warpgroup_mma(p16, vh, acc, is_async=True)
            acc, p16 = warpgroup_mma_wait(0, deps=[a_tok, p16])
        _gw_store(pacc_ptr, pml_ptr, b, hh, split, m_i, l_i, acc, HKV, S + 1, M, D, gl.SliceLayout(1, S_L), O_L)

    @gluon.jit
    def _gw_producer_cv(pt_row, kbase, vbase, k_ring, v_ring, vh0, vh1, ready, empty, lo, hi, ntiles,
                        PAGE: gl.constexpr, ROW_W: gl.constexpr, BN: gl.constexpr, C: gl.constexpr,
                        D: gl.constexpr, NS: gl.constexpr, PW: gl.constexpr, LAG: gl.constexpr):
        """_gw_producer that also converts V to FP16 (per-head halves), LAG tiles behind its gathers."""
        GLY: gl.constexpr = gl.BlockedLayout([1, 16], [4, 8], [PW, 1], [1, 0])
        VL: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [PW, 1], [1, 0])
        n_g = gl.arange(0, BN, layout=gl.SliceLayout(1, GLY))
        c_g = gl.arange(0, C, layout=gl.SliceLayout(0, GLY))
        t = gl.maximum(gl.minimum(lo + n_g, hi - 1), 0)
        pg = gl.load(pt_row + t // PAGE)
        for i in range(ntiles + LAG):
            if i < ntiles:
                s = i % NS
                off = ((pg.to(gl.int64) * PAGE + t % PAGE) * ROW_W)[:, None] + c_g[None, :]
                t = gl.maximum(gl.minimum(lo + (i + 1) * BN + n_g, hi - 1), 0)
                pg = gl.load(pt_row + t // PAGE)
                mbarrier.wait(empty.index(s), ((i // NS) & 1) ^ 1)
                cp.async_copy_global_to_shared(k_ring.index(s), kbase + off)
                cp.async_copy_global_to_shared(v_ring.index(s), vbase + off)
            cp.commit_group()
            if i >= LAG:
                sj = (i - LAG) % NS
                cp.wait_group(LAG)
                gl.barrier()
                vt = v_ring.index(sj)
                vh0.index(sj).store(vt.slice(0, D, 1).load(VL).to(gl.float16))
                vh1.index(sj).store(vt.slice(D, D, 1).load(VL).to(gl.float16))
                fence_async_shared()
                gl.barrier()
                mbarrier.arrive(ready.index(sj))

    @gluon.jit
    def _gw_consumer_cv(q, k_ring, vh, ready, empty, lo, hi, ntiles, scale, pacc_ptr, pml_ptr, b, hh, split,
                        HEAD: gl.constexpr, HKV: gl.constexpr, S: gl.constexpr, M: gl.constexpr, D: gl.constexpr,
                        BN: gl.constexpr, NS: gl.constexpr, PEEL: gl.constexpr):
        S_L: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1],
                                                      instr_shape=[16, BN, 32])
        O_L: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1],
                                                      instr_shape=[16, D, 16])
        P_L: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=O_L, k_width=2)
        m_i = gl.full([M], -1.0e30, gl.float32, layout=gl.SliceLayout(1, S_L))
        l_i = gl.zeros([M], gl.float32, layout=gl.SliceLayout(1, S_L))
        acc = gl.zeros([M, D], gl.float32, layout=O_L)
        n_s = gl.arange(0, BN, layout=gl.SliceLayout(0, S_L))
        for i in range(ntiles):
            s = i % NS
            mbarrier.wait(ready.index(s), (i // NS) & 1)
            s_tok = warpgroup_mma(q, k_ring.index(s).slice(HEAD * D, D, 1).permute((1, 0)),
                                  gl.zeros([M, BN], gl.float32, S_L), use_acc=False, is_async=True)
            sc = warpgroup_mma_wait(0, deps=[s_tok])
            if PEEL:
                # Only a chunk's last tile can run past its keys.
                if lo + (i + 1) * BN > hi:
                    sc = gl.where((lo + i * BN + n_s < hi)[None, :], sc, float("-inf"))
            else:
                sc = gl.where((lo + i * BN + n_s < hi)[None, :], sc, float("-inf"))
            m_new = gl.maximum(m_i, gl.max(sc, 1) * scale)
            alpha = gl.exp2(m_i - m_new)
            p = gl.exp2(gl.fma(sc, scale, -m_new[:, None]))
            l_i = l_i * alpha + gl.sum(p, 1)
            m_i = m_new
            acc = acc * gl.convert_layout(alpha, gl.SliceLayout(1, O_L))[:, None]
            p16 = gl.convert_layout(p.to(gl.float16), P_L)
            a_tok = warpgroup_mma(p16, vh.index(s), acc, is_async=True)
            acc, p16 = warpgroup_mma_wait(0, deps=[a_tok, p16])
            gl.barrier()
            mbarrier.arrive(empty.index(s))
        _gw_store(pacc_ptr, pml_ptr, b, hh, split, m_i, l_i, acc, HKV, S + 1, M, D, gl.SliceLayout(1, S_L), O_L)

    @gluon.jit
    def _gw_verify(qkv_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, pt_ptr, seq_ptr, loc_ptr, pacc_ptr, pml_ptr, qk_scale,
                   HQ: gl.constexpr, HKV: gl.constexpr, G: gl.constexpr, L: gl.constexpr, D: gl.constexpr,
                   PAGE: gl.constexpr, MAX_PAGES: gl.constexpr, S: gl.constexpr, BN: gl.constexpr,
                   NS: gl.constexpr, PW: gl.constexpr, PREGS: gl.constexpr, CREGS: gl.constexpr,
                   LAG: gl.constexpr = 0, PEEL: gl.constexpr = False):
        M: gl.constexpr = L * G
        C: gl.constexpr = 2 * D
        split = gl.program_id(0)
        b = gl.program_id(1)
        h = gl.program_id(2) * 2
        # Let the merge kernel launch now (programmatic dependent launch); it waits for this grid.
        gl.inline_asm_elementwise("griddepcontrol.launch_dependents;", "=r", [], dtype=gl.int32, is_pure=False, pack=1)
        QKV_W: gl.constexpr = (2 * HQ + 2 * HKV) * D
        ROW_W: gl.constexpr = HKV * D
        QL: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
        SH8: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=8, rank=2)
        SH16: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
        VSH: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
        SQ: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=64, element_bitwidth=8, rank=2)

        seq = gl.load(seq_ptr + b)
        # Each request's cached keys in S equal tile-aligned chunks.
        chunk = gl.cdiv(gl.cdiv(seq, S), BN) * BN
        lo = split * chunk
        hi = gl.minimum(lo + chunk, seq)
        ntiles = gl.cdiv(gl.maximum(hi - lo, 0), BN)
        pt_row = pt_ptr + b.to(gl.int64) * MAX_PAGES

        k_ring = gl.allocate_shared_memory(gl.float8e4nv, [NS, BN, C], SH8)
        v_ring = gl.allocate_shared_memory(gl.float8e4nv, [NS, BN, C], VSH)
        if LAG:
            vh0 = gl.allocate_shared_memory(gl.float16, [NS, BN, D], SH16)
            vh1 = gl.allocate_shared_memory(gl.float16, [NS, BN, D], SH16)
        else:
            vh0 = gl.allocate_shared_memory(gl.float16, [BN, D], SH16)
            vh1 = gl.allocate_shared_memory(gl.float16, [BN, D], SH16)
        ready = gl.allocate_shared_memory(gl.int64, [NS, 1], mbarrier.MBarrierLayout())
        empty = gl.allocate_shared_memory(gl.int64, [NS, 1], mbarrier.MBarrierLayout())
        for j in gl.static_range(NS):
            mbarrier.init(ready.index(j), count=1 if LAG else PW * 32)
            mbarrier.init(empty.index(j), count=2)
        r = gl.arange(0, M, layout=gl.SliceLayout(1, QL))
        cq = gl.arange(0, D, layout=gl.SliceLayout(0, QL))
        q_off = (b * L + r // G)[:, None] * QKV_W + (h * G + r % G)[:, None] * D + cq[None, :]
        x0 = gl.load(qkv_ptr + q_off).to(gl.float32)
        x0 = (x0 * gl.rsqrt(gl.sum(x0 * x0, 1) / D + 1e-5)[:, None]).to(gl.float8e4nv)
        q0 = gl.allocate_shared_memory(gl.float8e4nv, [M, D], SQ, x0)
        x1 = gl.load(qkv_ptr + q_off + G * D).to(gl.float32)
        x1 = (x1 * gl.rsqrt(gl.sum(x1 * x1, 1) / D + 1e-5)[:, None]).to(gl.float8e4nv)
        q1 = gl.allocate_shared_memory(gl.float8e4nv, [M, D], SQ, x1)
        scale = qk_scale * gl.load(ks_ptr)
        fence_async_shared()
        gl.barrier()
        if LAG:
            gl.warp_specialize([
                (_gw_consumer_cv, (q0, k_ring, vh0, ready, empty, lo, hi, ntiles, scale, pacc_ptr, pml_ptr, b, h,
                                   split, 0, HKV, S, M, D, BN, NS, PEEL)),
                (_gw_consumer_cv, (q1, k_ring, vh1, ready, empty, lo, hi, ntiles, scale, pacc_ptr, pml_ptr, b, h + 1,
                                   split, 1, HKV, S, M, D, BN, NS, PEEL)),
                (_gw_producer_cv, (pt_row, kc_ptr + h * D, vc_ptr + h * D, k_ring, v_ring, vh0, vh1, ready, empty,
                                   lo, hi, ntiles, PAGE, ROW_W, BN, C, D, NS, PW, LAG)),
            ], [4, PW], [CREGS, PREGS])
        else:
          gl.warp_specialize([
            (_gw_consumer, (q0, k_ring, v_ring, vh0, ready, empty, lo, hi, ntiles, scale, pacc_ptr, pml_ptr, b, h,
                            split, qkv_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, loc_ptr, 0, HQ, HKV, G, L, S, M, D, BN,
                            NS)),
            (_gw_consumer, (q1, k_ring, v_ring, vh1, ready, empty, lo, hi, ntiles, scale, pacc_ptr, pml_ptr, b,
                            h + 1, split, qkv_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, loc_ptr, 1, HQ, HKV, G, L, S, M, D,
                            BN, NS)),
            (_gw_producer, (pt_row, kc_ptr + h * D, vc_ptr + h * D, k_ring, v_ring, ready, empty, lo, hi, ntiles,
                            PAGE, ROW_W, BN, C, NS, PW)),
        ], [4, PW], [CREGS, PREGS])
        # This grid was launched alongside _new_tokens (which needs nothing from it); finish only after
        # it, so the merge (waiting on this grid) sees slot S too.
        gl.inline_asm_elementwise("griddepcontrol.wait;", "=r", [], dtype=gl.int32, is_pure=False, pack=1)


@triton.jit
def _norm_rope(ptr, off, ok, c, pos, cs_ptr, ROPE: tl.constexpr, D: tl.constexpr, HPP: tl.constexpr):
    """[R, HPP*D] rows of HPP heads at ptr + off: per-head rmsnorm (no weight), then interleaved
    RoPE at pos [R]. Masked-out entries are 0."""
    x = tl.load(ptr + off, mask=ok, other=0.0).to(tl.float32)
    sq = x * x
    if HPP == 1:
        r = tl.rsqrt(tl.sum(sq, 1) / D + 1e-5)[:, None]
    else:
        first = (c < D)[None, :]
        r0 = tl.rsqrt(tl.sum(tl.where(first, sq, 0.0), 1) / D + 1e-5)
        r1 = tl.rsqrt(tl.sum(tl.where(first, 0.0, sq), 1) / D + 1e-5)
        r = tl.where(first, r0[:, None], r1[:, None])
    if ROPE:
        xp = tl.load(ptr + (off ^ 1), mask=ok, other=0.0).to(tl.float32)
        cs_off = pos[:, None] * D + ((c % D) // 2)[None, :]
        cos = tl.load(cs_ptr + cs_off, mask=ok, other=1.0)
        sin = tl.load(cs_ptr + cs_off + D // 2, mask=ok, other=0.0)
        sign = tl.where(c % 2 == 0, -1.0, 1.0)
        x = x * cos + sign[None, :] * xp * sin
    return x * r


@triton.jit
def _tile(t, hi, pt_row, kc_ptr, vc_ptr, q, pos, c, m_i, l_i, acc, scale, hp0, MASKED: tl.constexpr,
          PAGE: tl.constexpr, ROW_W: tl.constexpr, D: tl.constexpr, WINDOW: tl.constexpr, QK8: tl.constexpr,
          DOT: tl.constexpr):
    """One tile of cached keys at positions t through the online softmax. MASKED: some t >= hi."""
    if MASKED:
        # Past the end, gather the last valid row again (no load masks); the scores are masked.
        t_ok = t < hi
        t_ld = tl.minimum(t, hi - 1)
    else:
        t_ld = t
    page = tl.load(pt_row + t_ld // PAGE)
    row = (page.to(tl.int64) * PAGE + t_ld % PAGE) * ROW_W + hp0 * D
    k = tl.load(kc_ptr + row[:, None] + c[None, :])
    v = tl.load(vc_ptr + row[:, None] + c[None, :])
    s = tl.dot(q, tl.trans(k if QK8 else k.to(DOT)))
    if MASKED:
        ok = t_ok[None, :]
        if WINDOW > 0:
            ok = ok & (t[None, :] > pos[:, None] - WINDOW)
        s = tl.where(ok, s, float("-inf"))
    elif WINDOW > 0:
        s = tl.where(t[None, :] > pos[:, None] - WINDOW, s, float("-inf"))
    # The scale (> 0) goes into the row maxima and the exponent's FMA, not every score.
    m_new = tl.maximum(m_i, tl.max(s, 1) * scale)
    alpha = tl.exp2(m_i - m_new)
    p = tl.exp2(tl.fma(s, scale, -m_new[:, None]))
    l_i = l_i * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None] + tl.dot(p.to(DOT), v.to(DOT))
    return m_new, l_i, acc


@triton.jit
def _new(split, b, seq, j, q, m_i, l_i, acc, hp0, c, scale, k_scale, qkv_ptr, cs_ptr, kc_ptr,
         vc_ptr, vs_ptr, loc_ptr, HQ: tl.constexpr, HKV: tl.constexpr, HPP: tl.constexpr,
         L: tl.constexpr, L_PAD: tl.constexpr, MG: tl.constexpr, D: tl.constexpr, ROPE: tl.constexpr,
         S: tl.constexpr, QK8: tl.constexpr, DOT: tl.constexpr):
    """Split 0 attends the block's own tokens (and stores them in the cache). Done before the
    cached keys, so these loads overlap the page-table and cache trips instead of following them."""
    QKV_W: tl.constexpr = (2 * HQ + 2 * HKV) * D
    ROW_W: tl.constexpr = HKV * D
    if split == 0:
        # The block's own tokens: normalize, store as FP8, attend causally.
        jn = tl.arange(0, L_PAD)
        jn_ok = jn < L
        tok = b * L + jn
        kn_off = (tok * QKV_W + (HQ + hp0) * D)[:, None] + c[None, :]
        kn = _norm_rope(qkv_ptr, kn_off, jn_ok[:, None], c, seq + jn, cs_ptr, ROPE, D, HPP)
        vn = tl.load(qkv_ptr + kn_off + HKV * D, mask=jn_ok[:, None], other=0.0)
        kq = (kn / k_scale).to(tl.float8e4nv)
        vq = (vn.to(tl.float32) / tl.load(vs_ptr)).to(tl.float8e4nv)
        slot = tl.load(loc_ptr + tok, mask=jn_ok, other=0) * ROW_W + hp0 * D
        tl.store(kc_ptr + slot[:, None] + c[None, :], kq, mask=jn_ok[:, None])
        tl.store(vc_ptr + slot[:, None] + c[None, :], vq, mask=jn_ok[:, None])
        s = tl.dot(q, tl.trans(kq if QK8 else kq.to(DOT))) * scale
        ok = jn_ok[None, :] & (jn[None, :] <= j[:, None])
        s = tl.where(ok, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(DOT), vq.to(DOT))
        m_i = m_new

    return m_i, l_i, acc


@triton.jit
def _finish(split, b, j, m_i, l_i, acc, hp0, hp, rr, r_ok, q_ok, c, pacc_ptr, pml_ptr, HKV: tl.constexpr,
            MG: tl.constexpr, D: tl.constexpr, S: tl.constexpr):
    """Every split writes its partial (max, sum, accumulator) for KV heads hp0 .. hp0 + HPP - 1."""
    # Partials [B, HKV, S, MG]: each row's own head's D columns of acc.
    part = ((b * HKV + hp0 + hp) * S + split) * MG + rr
    tl.store(pml_ptr + part * 2, m_i, mask=r_ok)
    tl.store(pml_ptr + part * 2 + 1, l_i, mask=r_ok)
    tl.store(pacc_ptr + part[:, None] * D + (c % D)[None, :], acc, mask=q_ok)


@triton.jit
def _attn_split(qkv_ptr, cs_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, pt_ptr, seq_ptr, loc_ptr,
                pacc_ptr, pml_ptr, cnt_ptr, out_ptr, qk_scale,
                HQ: tl.constexpr, HKV: tl.constexpr, HPP: tl.constexpr, G: tl.constexpr, L: tl.constexpr,
                MG: tl.constexpr, M_PAD: tl.constexpr, L_PAD: tl.constexpr, D: tl.constexpr,
                PAGE: tl.constexpr, MAX_PAGES: tl.constexpr, WINDOW: tl.constexpr, ROPE: tl.constexpr,
                S: tl.constexpr, CHUNK: tl.constexpr, BLOCK_N: tl.constexpr, STAGES: tl.constexpr,
                FUSE: tl.constexpr, S_PAD: tl.constexpr, QK8: tl.constexpr, PDL: tl.constexpr,
                DOT: tl.constexpr, PEEL: tl.constexpr = False, DUAL: tl.constexpr = False,
                BALANCE: tl.constexpr = False):
    """One program: a split of request b's context for HPP adjacent KV heads, which share each
    cache row's contiguous HPP*D bytes. Queries are block-diagonal [HPP*MG, HPP*D] (row r is KV
    head hp = r // MG of the pair, and holds values only in that head's D columns), so one
    [rows, HPP*D] x [HPP*D, N] product scores every head against its own keys."""
    split = tl.program_id(0)
    b = tl.program_id(1)
    # DUAL (with HPP 1): the program also runs KV head hp0 + 1 as a second, independent chain.
    hp0 = tl.program_id(2) * (2 if DUAL else HPP)
    if PDL:
        # Let the merge kernel launch now; it waits on-device for this grid to finish.
        gdc_launch_dependents()
    QKV_W: tl.constexpr = (2 * HQ + 2 * HKV) * D
    ROW_W: tl.constexpr = HKV * D
    C: tl.constexpr = HPP * D

    seq = tl.load(seq_ptr + b)
    c = tl.arange(0, C)
    r = tl.arange(0, M_PAD)
    hp = r // MG
    rr = r % MG  # token rr // G, query head (hp0 + hp) * G + rr % G
    r_ok = r < HPP * MG
    j = rr // G
    pos = seq + j
    q_off = (b * L + j)[:, None] * QKV_W + ((hp0 + hp) * G + rr % G)[:, None] * D + (c % D)[None, :]
    q_ok = r_ok[:, None] & ((c // D)[None, :] == hp[:, None])
    q = _norm_rope(qkv_ptr, q_off, q_ok, c, pos, cs_ptr, ROPE, D, HPP)
    if QK8:
        # Queries rounded to E4M3 (as FA3 does with an FP8 cache): QK^T runs on the FP8 cache as is.
        q = q.to(tl.float8e4nv)
    else:
        q = q.to(DOT)
    q1 = q
    if DUAL:
        q1 = _norm_rope(qkv_ptr, q_off + G * D, q_ok, c, pos, cs_ptr, ROPE, D, HPP)
        q1 = q1.to(tl.float8e4nv) if QK8 else q1.to(DOT)
    k_scale = tl.load(ks_ptr)
    scale = qk_scale * k_scale  # log2 units, applied to q . k_fp8

    m_i = tl.full([M_PAD], -1.0e30, tl.float32)
    l_i = tl.zeros([M_PAD], tl.float32)
    acc = tl.zeros([M_PAD, C], tl.float32)
    m1, l1, acc1 = m_i, l_i, acc
    m_i, l_i, acc = _new(split, b, seq, j, q, m_i, l_i, acc, hp0, c, scale, k_scale, qkv_ptr, cs_ptr, kc_ptr,
                         vc_ptr, vs_ptr, loc_ptr, HQ, HKV, HPP, L, L_PAD, MG, D, ROPE, S, QK8, DOT)
    if DUAL:
        m1, l1, acc1 = _new(split, b, seq, j, q1, m1, l1, acc1, hp0 + 1, c, scale, k_scale, qkv_ptr, cs_ptr,
                            kc_ptr, vc_ptr, vs_ptr, loc_ptr, HQ, HKV, HPP, L, L_PAD, MG, D, ROPE, S, QK8, DOT)

    if WINDOW > 0:
        base = tl.maximum(seq - (WINDOW - 1), 0)
    else:
        base = seq * 0
    if BALANCE:
        # Each request's own context in S equal tile-aligned chunks: every program does the same work.
        chunk = tl.cdiv(tl.cdiv(seq - base, S), BLOCK_N) * BLOCK_N
    else:
        chunk = CHUNK
    lo = base + split * chunk
    hi = tl.minimum(lo + chunk, seq)
    pt_row = pt_ptr + b.to(tl.int64) * MAX_PAGES
    n = tl.arange(0, BLOCK_N)
    # Whole tiles need no bounds masks; full layers need no masks at all there. The tail is masked.
    # Only for single-token decode: with a verify block's 64 rows the second copy costs registers.
    if L == 1 or PEEL:
        full_end = lo + tl.maximum(hi - lo, 0) // BLOCK_N * BLOCK_N
    else:
        full_end = lo
    for start in tl.range(lo, full_end, BLOCK_N, num_stages=STAGES):
        m_i, l_i, acc = _tile(start + n, hi, pt_row, kc_ptr, vc_ptr, q, pos, c, m_i, l_i, acc, scale, hp0,
                              False, PAGE, ROW_W, D, WINDOW, QK8, DOT)
        if DUAL:
            m1, l1, acc1 = _tile(start + n, hi, pt_row, kc_ptr, vc_ptr, q1, pos, c, m1, l1, acc1, scale, hp0 + 1,
                                 False, PAGE, ROW_W, D, WINDOW, QK8, DOT)
    for start in tl.range(full_end, hi, BLOCK_N, num_stages=STAGES):
        m_i, l_i, acc = _tile(start + n, hi, pt_row, kc_ptr, vc_ptr, q, pos, c, m_i, l_i, acc, scale, hp0,
                              True, PAGE, ROW_W, D, WINDOW, QK8, DOT)
        if DUAL:
            m1, l1, acc1 = _tile(start + n, hi, pt_row, kc_ptr, vc_ptr, q1, pos, c, m1, l1, acc1, scale, hp0 + 1,
                                 True, PAGE, ROW_W, D, WINDOW, QK8, DOT)

    _finish(split, b, j, m_i, l_i, acc, hp0, hp, rr, r_ok, q_ok, c, pacc_ptr, pml_ptr, HKV, MG, D, S)
    if DUAL:
        _finish(split, b, j, m1, l1, acc1, hp0 + 1, hp, rr, r_ok, q_ok, c, pacc_ptr, pml_ptr, HKV, MG, D, S)

    if FUSE:
        # The last of this (request, head pair)'s S programs to finish merges the partials and
        # writes the output, then resets the counter for the next call.
        cnt = cnt_ptr + b * (HKV // HPP) + hp0 // HPP
        if tl.atomic_add(cnt, 1, sem="acq_rel") == S - 1:
            first = ((b * HKV + hp0 + hp) * S) * MG + rr
            # All splits' maxima and sums at once: [S_PAD, M_PAD].
            si = tl.arange(0, S_PAD)
            at = first[None, :] + si[:, None] * MG
            ok2 = (si < S)[:, None] & r_ok[None, :]
            m = tl.load(pml_ptr + at * 2, mask=ok2, other=-1.0e30, cache_modifier=".cg")
            m_max = tl.max(m, 0)
            denom = tl.sum(tl.exp2(m - m_max[None, :]) * tl.load(pml_ptr + at * 2 + 1, mask=ok2, other=0.0,
                                                                  cache_modifier=".cg"), 0)
            # Unrolled, so every split's accumulator load is in flight at once.
            o = tl.zeros([M_PAD, C], tl.float32)
            for i in tl.static_range(S):
                at1 = first + i * MG
                w = tl.exp2(tl.load(pml_ptr + at1 * 2, mask=r_ok, other=-1.0e30, cache_modifier=".cg") - m_max)
                o += w[:, None] * tl.load(pacc_ptr + at1[:, None] * D + (c % D)[None, :], mask=q_ok, other=0.0,
                                          cache_modifier=".cg")
            _norm_gate_store(o / tl.where(r_ok, denom, 1.0)[:, None] * tl.load(vs_ptr), qkv_ptr, out_ptr,
                             b * L + j, (hp0 + hp) * G + rr % G, c, q_ok, HQ, HKV, D)
            tl.atomic_xchg(cnt, 0)


@triton.jit
def _new_tokens(qkv_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, seq_ptr, loc_ptr, pacc_ptr, pml_ptr, qk_scale,
                HQ: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr, L: tl.constexpr, MG: tl.constexpr,
                L_PAD: tl.constexpr, D: tl.constexpr, S: tl.constexpr, DOT: tl.constexpr):
    """For _gw_verify: one program per (request, KV head) stores the block's new keys and values as FP8
    and attends them causally, into partial slot S (of S + 1)."""
    b = tl.program_id(0)
    hp0 = tl.program_id(1)
    gdc_launch_dependents()
    QKV_W: tl.constexpr = (2 * HQ + 2 * HKV) * D
    seq = tl.load(seq_ptr + b)
    c = tl.arange(0, D)
    rr = tl.arange(0, MG)
    j = rr // G
    q_off = (b * L + j)[:, None] * QKV_W + (hp0 * G + rr % G)[:, None] * D + c[None, :]
    q_ok = rr[:, None] < MG
    q = _norm_rope(qkv_ptr, q_off, q_ok, c, seq + j, qkv_ptr, False, D, 1).to(tl.float8e4nv)
    k_scale = tl.load(ks_ptr)
    m_i = tl.full([MG], -1.0e30, tl.float32)
    l_i = tl.zeros([MG], tl.float32)
    acc = tl.zeros([MG, D], tl.float32)
    m_i, l_i, acc = _new(0, b, seq, j, q, m_i, l_i, acc, hp0, c, qk_scale * k_scale, k_scale, qkv_ptr, qkv_ptr,
                         kc_ptr, vc_ptr, vs_ptr, loc_ptr, HQ, HKV, 1, L, L_PAD, MG, D, False, S, True, DOT)
    _finish(S, b, j, m_i, l_i, acc, hp0, 0, rr, rr < MG, q_ok, c, pacc_ptr, pml_ptr, HKV, MG, D, S + 1)


@triton.jit
def _norm_gate_store(o, qkv_ptr, out_ptr, tok, head, c, ok, HQ: tl.constexpr, HKV: tl.constexpr,
                     D: tl.constexpr):
    """out[tok, head] = rmsnorm(o) * sigmoid(gate) for [R, C] rows holding one head's D values each
    (the rest masked out and 0)."""
    QKV_W: tl.constexpr = (2 * HQ + 2 * HKV) * D
    o = o * tl.rsqrt(tl.sum(o * o, 1) / D + 1e-5)[:, None]
    col = head[:, None] * D + (c % D)[None, :]
    gate = tl.load(qkv_ptr + tok[:, None] * QKV_W + (HQ + 2 * HKV) * D + col, mask=ok, other=0.0).to(tl.float32)
    tl.store(out_ptr + tok[:, None] * HQ * D + col, (o * tl.sigmoid(gate)).to(tl.bfloat16), mask=ok)


@triton.jit
def _combine(qkv_ptr, pacc_ptr, pml_ptr, vs_ptr, out_ptr,
             HQ: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr, L: tl.constexpr, M: tl.constexpr,
             D: tl.constexpr, S: tl.constexpr, S_PAD: tl.constexpr, SB: tl.constexpr, PDL: tl.constexpr):
    tok = tl.program_id(0)
    head = tl.program_id(1)
    b = tok // L
    h = head // G
    QKV_W: tl.constexpr = (2 * HQ + 2 * HKV) * D
    d = tl.arange(0, D)
    gate = tl.load(qkv_ptr + tok * QKV_W + (HQ + 2 * HKV) * D + head * D + d).to(tl.float32)
    if PDL:
        # Launched early (programmatic dependent launch): wait here for _attn_split's partials.
        gdc_wait()
    # This head's row in each split's partial: [S] maxima and sums, [S, D] accumulators.
    row = (b * HKV + h) * S * M + (tok % L) * G + head % G
    s = tl.arange(0, S_PAD)
    m = tl.load(pml_ptr + (row + s * M) * 2, mask=s < S, other=-1.0e30)
    l = tl.load(pml_ptr + (row + s * M) * 2 + 1, mask=s < S, other=0.0)
    m_max = tl.max(m, 0)
    denom = tl.sum(tl.exp2(m - m_max) * l, 0)
    acc = tl.zeros([D], tl.float32)
    sb = tl.arange(0, SB)
    for i in range(0, S, SB):
        si = i + sb
        w = tl.exp2(tl.load(pml_ptr + (row + si * M) * 2, mask=si < S, other=-1.0e30) - m_max)
        a = tl.load(pacc_ptr + (row + si * M)[:, None] * D + d[None, :], mask=(si < S)[:, None], other=0.0)
        acc += tl.sum(w[:, None] * a, 0)
    o = acc / denom * tl.load(vs_ptr)
    o = o * tl.rsqrt(tl.sum(o * o, 0) / D + 1e-5)
    tl.store(out_ptr + tok * HQ * D + head * D + d, (o * tl.sigmoid(gate)).to(tl.bfloat16))


@triton.jit
def _combine_hb(qkv_ptr, pacc_ptr, pml_ptr, vs_ptr, out_ptr,
                HQ: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr, L: tl.constexpr, M: tl.constexpr,
                D: tl.constexpr, S: tl.constexpr, S_PAD: tl.constexpr, SB: tl.constexpr, HB: tl.constexpr,
                PDL: tl.constexpr):
    """_combine for HB adjacent query heads of one KV head per program: their partial rows are
    contiguous in every split, so each split's [HB, D] block is one contiguous read. Faster on
    sliding layers (fewer, fuller programs); full layers keep _combine."""
    tok = tl.program_id(0)
    head0 = tl.program_id(1) * HB
    b = tok // L
    h = head0 // G
    QKV_W: tl.constexpr = (2 * HQ + 2 * HKV) * D
    d = tl.arange(0, D)
    hb = tl.arange(0, HB)
    col = (head0 + hb)[:, None] * D + d[None, :]
    gate = tl.load(qkv_ptr + tok * QKV_W + (HQ + 2 * HKV) * D + col).to(tl.float32)
    if PDL:
        gdc_wait()
    row = (b * HKV + h) * S * M + (tok % L) * G + head0 % G + hb  # [HB] rows in split 0
    s = tl.arange(0, S_PAD)
    at = row[None, :] + (s * M)[:, None]
    m = tl.load(pml_ptr + at * 2, mask=(s < S)[:, None], other=-1.0e30)
    l = tl.load(pml_ptr + at * 2 + 1, mask=(s < S)[:, None], other=0.0)
    m_max = tl.max(m, 0)
    denom = tl.sum(tl.exp2(m - m_max[None, :]) * l, 0)
    acc = tl.zeros([HB, D], tl.float32)
    sb = tl.arange(0, SB)
    for i in range(0, S, SB):
        si = i + sb
        at1 = row[None, :] + (si * M)[:, None]
        ok = (si < S)[:, None]
        w = tl.exp2(tl.load(pml_ptr + at1 * 2, mask=ok, other=-1.0e30) - m_max[None, :])
        a = tl.load(pacc_ptr + at1[:, :, None] * D + d[None, None, :], mask=ok[:, :, None], other=0.0)
        acc += tl.sum(w[:, :, None] * a, 0)
    o = acc / denom[:, None] * tl.load(vs_ptr)
    o = o * tl.rsqrt(tl.sum(o * o, 1) / D + 1e-5)[:, None]
    tl.store(out_ptr + tok * HQ * D + col, (o * tl.sigmoid(gate)).to(tl.bfloat16))


def _plan(cfg, max_pages, device):
    """Launch parameters from shapes only, tuned on an H100 over the scored shapes: splits per
    (request, KV head group), tile size, warps, pipeline stages, KV heads per program (hpp),
    whether the last program merges the splits (fuse), and whether QK^T runs in FP8 (qk8:
    queries rounded to E4M3, as FA3 does with an FP8 cache)."""
    batch, length, hkv = cfg["batch"], cfg["q_len"], cfg["num_kv_heads"]
    heads = batch * hkv
    if cfg["window"]:
        # Sliding layers: at most 2,047 cached keys; latency-bound.
        # The merge takes 2 query heads per one-warp program (hb, cw) and every split in one pass (sb):
        # each further pass is another serial L2 round trip on the critical path.
        splits = 32 if heads <= 4 else (16 if heads <= 32 and length == 1 else 8)
        if heads <= 32:
            return dict(splits=splits, block_n=64, warps=4, stages=5 if length > 1 and heads > 4 else 3,
                        hpp=1, fuse=False, qk8=True, hb=2, sb=splits, cw=1)
        # Large batches: smaller, deeply pipelined programs.
        return dict(splits=splits, block_n=32, warps=2 if length == 1 else 4, stages=5, hpp=1, fuse=False,
                    qk8=True, peel=length > 1, hb=2, sb=splits, cw=1)
    if heads <= (2 if length == 1 else 4):
        return dict(splits=128, block_n=128, warps=4, stages=6, hpp=1, fuse=False, qk8=True)
    if length == 1:
        # Many small one-warp programs with deep pipelines keep the most scattered row reads in flight.
        return dict(splits=max(8, 2048 // heads), block_n=16, warps=1, stages=5 if heads <= 32 else 6, hpp=2,
                    fuse=False, qk8=True)
    # Verify blocks: each program runs both KV heads as two independent 64-row chains (dual), which
    # hides latency better than more programs; 3 fit per SM (160 registers), or 4 when capped at 128
    # (no spills). The splits fill exactly one wave, and each request's context is cut into equal
    # chunks (balance) so none idles. 4 per SM wins from 6 pairs up (batch 32: 303 -> 292 us, 64:
    # 593 -> 568 on an H100); below that only where it fills the wave better (batch 3-4 lose 2-4%).
    pairs = batch * max(1, hkv // 2)
    sms = _num_sms(device)
    if HAVE_GLUON and pairs >= 6 and hkv % 2 == 0 and length * cfg["num_q_heads"] // hkv == 64 \
            and cfg["head_dim"] == 64:
        # The warp-specialized Gluon kernel: 2 programs per SM (384 threads). One wave of programs unless
        # two fill the slots clearly better (H100, ctx 80K: batch 16 155 -> 145 us with 16 splits instead
        # of 32, batch 12 124 -> 116; batch 40 keeps two waves, 344 vs 332).
        slots = 2 * sms
        fill = {w: (w * slots // pairs) * pairs / (w * slots) for w in (1, 2)}
        waves = 2 if fill[2] > fill[1] + 0.05 else 1
        return dict(gluon=True, splits=max(2, waves * slots // pairs), block_n=64, stages=3, lag=1, gpeel=True, pw=4,
                    pregs=40, cregs=96, maxnreg=80)
    fill = {per: max(4, per * sms // pairs) * pairs / (per * sms) for per in (3, 4)}
    per = 4 if pairs >= 6 or fill[4] > fill[3] else 3
    return dict(splits=max(4, per * sms // pairs), block_n=32, warps=4, stages=5, hpp=1, fuse=False,
                qk8=True, peel=True, dual=True, balance=True, maxnreg=128 if per == 4 else None)


def kernel_fn(data: Inputs) -> torch.Tensor:
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    group = hq // hkv
    qkv = data["qkv"]
    device = qkv.device
    page_table = data["page_table"]
    max_pages = page_table.shape[1]
    # The native single-tile grid wins where at most sixteen KV heads are processed.
    # Keep the established Triton plans on larger batches and verify blocks.
    if (cfg["window"] == 2048 and length == 1 and batch * hkv <= 16
            and group == 8 and d == 64 and cfg["page_size"] in (1, 64) and cfg["rope_theta"]):
        splits = min(32, triton.cdiv(max_pages * cfg["page_size"], 64))
        pacc = torch.empty((batch, hkv, splits, 8, 64), dtype=torch.float32, device=device)
        pml = torch.empty((batch, hkv, splits, 8, 2), dtype=torch.float32, device=device)
        out = torch.empty((batch, 1, hq * 64), dtype=torch.bfloat16, device=device)
        cs = _cos_sin(cfg["rope_theta"], 64, device)
        _NATIVE_SLIDING.attention(qkv, cs, data["k_cache"], data["v_cache"], page_table,
                                 data["seq_lens"], data["cache_loc"], data["k_scale"],
                                 data["v_scale"], pacc, pml, cfg["page_size"],
                                 cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E, True)
        _combine_hb[(batch, hq // 2)](
            qkv, pacc, pml, data["v_scale"], out,
            HQ=hq, HKV=hkv, G=8, L=1, M=8, D=64, S=splits,
            S_PAD=triton.next_power_of_2(splits), SB=triton.next_power_of_2(splits),
            HB=2, PDL=USE_PDL, num_warps=1, launch_pdl=USE_PDL)
        return out
    plan = _plan(cfg, max_pages, device)
    if plan.get("gluon"):
        return _verify_gluon(data, cfg, plan)
    span = max_pages * cfg["page_size"]
    if cfg["window"]:
        span = min(span, cfg["window"])
    block_n = plan["block_n"]
    chunk = triton.cdiv(triton.cdiv(span, plan["splits"]), block_n) * block_n
    splits = plan["splits"] if plan.get("balance") else triton.cdiv(span, chunk)
    rows = length * group
    hpp = plan["hpp"] if hkv % plan["hpp"] == 0 else 1
    dual = plan.get("dual", False) and hpp == 1 and hkv % 2 == 0
    rope = bool(cfg["rope_theta"])
    cs = _cos_sin(cfg["rope_theta"], d, device) if rope else qkv

    pacc = torch.empty((batch, hkv, splits, rows, d), dtype=torch.float32, device=device)
    pml = torch.empty((batch, hkv, splits, rows, 2), dtype=torch.float32, device=device)
    out = torch.empty((batch, length, hq * d), dtype=torch.bfloat16, device=device)
    fuse = plan["fuse"] and splits <= 64 and not dual  # the fused merge unrolls over splits
    _attn_split[(splits, batch, hkv // (2 if dual else hpp))](
        qkv, cs, data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"], page_table,
        data["seq_lens"], data["cache_loc"], pacc, pml, _counters(batch * hkv, device), out,
        cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E,
        HQ=hq, HKV=hkv, HPP=hpp, G=group, L=length, MG=rows, M_PAD=max(16, triton.next_power_of_2(hpp * rows)),
        L_PAD=max(16, triton.next_power_of_2(length)), D=d, PAGE=cfg["page_size"], MAX_PAGES=max_pages,
        WINDOW=cfg["window"], ROPE=rope, S=splits, CHUNK=chunk, BLOCK_N=block_n, STAGES=plan["stages"],
        FUSE=fuse, S_PAD=triton.next_power_of_2(splits), QK8=plan["qk8"], PDL=USE_PDL and not fuse, DOT=DOT_DTYPE, PEEL=plan.get("peel", False), DUAL=dual, BALANCE=plan.get("balance", False), num_warps=plan["warps"],
        **({"maxnreg": plan["maxnreg"]} if plan.get("maxnreg") else {}))
    if fuse:
        return out
    # Eight warps independently merge up to 16 FP32 splits, then combine their
    # partial results in shared memory. Larger batches retain the Triton merge.
    if not cfg["window"] and batch * hkv <= 4 and 64 < splits <= 128:
        _NATIVE_MERGE.combine_full(qkv, pacc, pml, data["v_scale"], out,
                                   hq, hkv, length, splits, USE_PDL)
        return out
    hb = plan.get("hb", 1) if group % plan.get("hb", 1) == 0 else 1
    if hb > 1:
        sb = plan.get("sb", max(1, min(triton.next_power_of_2(splits), 32 // hb)))
        _combine_hb[(batch * length, hq // hb)](
            qkv, pacc, pml, data["v_scale"], out,
            HQ=hq, HKV=hkv, G=group, L=length, M=rows, D=d, S=splits, S_PAD=triton.next_power_of_2(splits),
            SB=sb, HB=hb, PDL=USE_PDL, num_warps=plan.get("cw", 4), launch_pdl=USE_PDL)
        return out
    _combine[(batch * length, hq)](
        qkv, pacc, pml, data["v_scale"], out,
        HQ=hq, HKV=hkv, G=group, L=length, M=rows, D=d, S=splits, S_PAD=triton.next_power_of_2(splits),
        # Up to 128 splits in one pass, so every partial's load is in flight at once.
        SB=min(128, triton.next_power_of_2(splits)), PDL=USE_PDL, num_warps=4 if splits > 32 else 2,
        launch_pdl=USE_PDL)
    return out


def _verify_gluon(data, cfg, plan):
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    group = hq // hkv
    rows = length * group
    qkv = data["qkv"]
    device = qkv.device
    page_table = data["page_table"]
    splits = plan["splits"]
    pacc = torch.empty((batch, hkv, splits + 1, rows, d), dtype=torch.float32, device=device)
    pml = torch.empty((batch, hkv, splits + 1, rows, 2), dtype=torch.float32, device=device)
    out = torch.empty((batch, length, hq * d), dtype=torch.bfloat16, device=device)
    qk_scale = cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E
    _new_tokens[(batch, hkv)](
        qkv, data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"], data["seq_lens"], data["cache_loc"],
        pacc, pml, qk_scale, HQ=hq, HKV=hkv, G=group, L=length, MG=rows, L_PAD=max(16, triton.next_power_of_2(length)),
        D=d, S=splits, DOT=DOT_DTYPE, num_warps=4)
    _gw_verify[(splits, batch, hkv // 2)](
        qkv, data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"], page_table, data["seq_lens"],
        data["cache_loc"], pacc, pml, qk_scale,
        HQ=hq, HKV=hkv, G=group, L=length, D=d, PAGE=cfg["page_size"], MAX_PAGES=page_table.shape[1], S=splits,
        BN=plan["block_n"], NS=plan["stages"], PW=plan["pw"], PREGS=plan["pregs"], CREGS=plan["cregs"],
        LAG=plan.get("lag", 0), PEEL=plan.get("gpeel", False), num_warps=4, maxnreg=plan["maxnreg"], launch_pdl=True)
    s = splits + 1
    _combine[(batch * length, hq)](
        qkv, pacc, pml, data["v_scale"], out,
        HQ=hq, HKV=hkv, G=group, L=length, M=rows, D=d, S=s, S_PAD=triton.next_power_of_2(s),
        SB=min(128, triton.next_power_of_2(s)), PDL=USE_PDL, num_warps=4 if s > 32 else 2, launch_pdl=USE_PDL)
    return out

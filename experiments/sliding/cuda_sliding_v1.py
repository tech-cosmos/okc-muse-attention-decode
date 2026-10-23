"""Native sliding decode experiment. Imported only by its local driver."""
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline

CPP = r'''
#include <torch/extension.h>
void attention(torch::Tensor qkv, torch::Tensor cs, torch::Tensor kc, torch::Tensor vc,
 torch::Tensor pt, torch::Tensor seq, torch::Tensor loc, torch::Tensor ks, torch::Tensor vs,
 torch::Tensor pa, torch::Tensor pm, torch::Tensor count, torch::Tensor out,
 int splits, int page, double scale, int stages, int warps, bool fused);
'''
CUDA = r'''
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <cuda/atomic>
using namespace nvcuda;
__device__ __forceinline__ float warp_sum(float x) {
 #pragma unroll
 for(int i=16;i;i/=2) x+=__shfl_xor_sync(0xffffffff,x,i);
 return x;
}
__device__ __forceinline__ unsigned char fp8(float x) {return __nv_cvt_float_to_fp8(x,__NV_SATFINITE,__NV_E4M3);}
__device__ __forceinline__ half fp16(unsigned char x) {return __nv_cvt_fp8_to_halfraw(x,__NV_E4M3);}
__device__ __forceinline__ void copy16(void* dst,const void* src) {
 asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"::"r"((unsigned)__cvta_generic_to_shared(dst)),"l"(src));
}
__device__ __forceinline__ float rotated(const __nv_bfloat16* p,const float* cs,int pos,int col) {
 float x=__bfloat162float(p[col]), y=__bfloat162float(p[col^1]);
 float c=cs[(long)pos*64+col/2],s=cs[(long)pos*64+32+col/2];
 return x*c+(col%2?1.f:-1.f)*y*s;
}
template<int STAGES,int W,int PAGE,bool FUSED>
__global__ __launch_bounds__(W*32) void sliding(const __nv_bfloat16* __restrict__ qkv,const float* __restrict__ cs,
 unsigned char* kc,unsigned char* vc,const int* __restrict__ pt,const int* seqs,const long* loc,
 const float* ks,const float* vs,float* pa,float* pm,int* counts,__nv_bfloat16* out,
 int HQ,int HKV,int MP,int S,float SCALE) {
 constexpr int N=16,D=64;
 __shared__ __align__(16) unsigned char raw[W][STAGES][2][N*D];
 __shared__ __align__(32) half kv[W][2][N*(D+8)];
 __shared__ __align__(32) half Q[16*(D+8)];
 __shared__ __align__(16) unsigned char own[2][D];
 __shared__ int last;
 const unsigned tid=threadIdx.x,warp=tid/32,lane=tid%32;
 const int b=blockIdx.y,h=blockIdx.z,split=blockIdx.x;
 const int seq=seqs[b],width=(2*HQ+2*HKV)*D;
 const int base=max(0,seq-2047),total=S*W;
 const int chunk=((seq-base+N*total-1)/(N*total))*N;
 const int ws=split*W+warp,lo=base+ws*chunk,hi=min(lo+chunk,seq);
 const int oldtiles=max(0,(hi-lo+N-1)/N),tiles=oldtiles+(ws==0);
 const float kscale=*ks,vscale=*vs;
 for(int row=warp;row<16;row+=W) {
   float x=0,y=0;
   if(row<8) {
     const __nv_bfloat16* p=qkv+b*width+(h*8+row)*64;
     x=__bfloat162float(p[lane]);y=__bfloat162float(p[lane+32]);
     float r=rsqrtf(warp_sum(x*x+y*y)/64.f+1e-5f);
     x=rotated(p,cs,seq,lane)*r; y=rotated(p,cs,seq,lane+32)*r;
   }
   Q[row*72+lane]=fp16(fp8(x));Q[row*72+lane+32]=fp16(fp8(y));
 }
 if(split==0 && warp==0) {
   const __nv_bfloat16* p=qkv+b*width+(HQ+h)*64;
   float x=__bfloat162float(p[lane]),y=__bfloat162float(p[lane+32]);
   float r=rsqrtf(warp_sum(x*x+y*y)/64.f+1e-5f)/kscale;
   unsigned char k0=fp8(rotated(p,cs,seq,lane)*r),k1=fp8(rotated(p,cs,seq,lane+32)*r);
   unsigned char v0=fp8(__bfloat162float(p[HKV*64+lane])/vscale),v1=fp8(__bfloat162float(p[HKV*64+lane+32])/vscale);
   own[0][lane]=k0;own[0][lane+32]=k1;own[1][lane]=v0;own[1][lane+32]=v1;
   long addr=loc[b]*HKV*64+h*64;
   kc[addr+lane]=k0;kc[addr+lane+32]=k1;vc[addr+lane]=v0;vc[addr+lane+32]=v1;
 }
 __syncthreads();
 wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> q[4];
 #pragma unroll
 for(int d=0;d<4;d++) wmma::load_matrix_sync(q[d],Q+d*16,72);
 wmma::fragment<wmma::accumulator,16,16,16,float> acc[4];
 #pragma unroll
 for(int n=0;n<4;n++)wmma::fill_fragment(acc[n],0.f);
 float mx=-1e30f,sum=0;
 for(int step=0;step<tiles+STAGES-1;step++) {
   if(step<tiles) {
     int stage=step%STAGES;
     bool isnew=step==oldtiles;
     if(!isnew) {
       #pragma unroll
       for(int i=lane;i<N*4;i+=32) {
         int t=min(lo+step*N+i/4,hi-1);
         long row=((long)pt[b*MP+t/PAGE]*PAGE+t%PAGE)*HKV*D+h*D;
         copy16(raw[warp][stage][0]+i*16,kc+row+(i%4)*16);
         copy16(raw[warp][stage][1]+i*16,vc+row+(i%4)*16);
       }
     } else {
       for(int i=lane;i<N*D;i+=32) {
         raw[warp][stage][0][i]=i<D?own[0][i]:0;
         raw[warp][stage][1][i]=i<D?own[1][i]:0;
       }
     }
     asm volatile("cp.async.commit_group;":::"memory");
   }
   if(step<STAGES-1)continue;
   const int tile=step-(STAGES-1),stage=tile%STAGES;
   if(step<tiles) {asm volatile("cp.async.wait_group %0;"::"n"(STAGES-1):"memory");}
   else {asm volatile("cp.async.wait_group 0;":::"memory");}
   __syncwarp();
   #pragma unroll
   for(unsigned i=lane*2;i<N*D;i+=64) {
     #pragma unroll
     for(int x=0;x<2;x++) {
       __half2_raw v=__nv_cvt_fp8x2_to_halfraw2(*(unsigned short*)(raw[warp][stage][x]+i),__NV_E4M3);
       *(half2*)(kv[warp][x]+(i/D)*72+i%D)=v;
     }
   }
   __syncwarp();
   wmma::fragment<wmma::accumulator,16,16,16,float> score;
   wmma::fill_fragment(score,0.f);
   #pragma unroll
   for(int d=0;d<4;d++) {
     wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::col_major> k;
     wmma::load_matrix_sync(k,kv[warp][0]+d*16,72);
     wmma::mma_sync(score,q[d],k,score);
   }
   int bound=tile==oldtiles?0:hi-lo-tile*N-1;
   float now=-1e30f;
   #pragma unroll
   for(int i=0;i<8;i++) {
     int col=(lane%4)*2+(i&1)+(i&4)*2;
     if(col>bound)score.x[i]=-1e30f;
     if(!(i&2))now=fmaxf(now,score.x[i]);
   }
   now=fmaxf(now,__shfl_xor_sync(0xffffffff,now,1));
   now=fmaxf(now,__shfl_xor_sync(0xffffffff,now,2));
   now=fmaxf(now*SCALE*kscale,mx);float a=exp2f(mx-now);mx=now;
   float add=0;
   #pragma unroll
   for(int i=0;i<8;i++) {
     float p=exp2f(fmaf(score.x[i],SCALE*kscale,-mx));
     if(!(i&2))add+=p;
     score.x[i]=p;
   }
   add+=__shfl_xor_sync(0xffffffff,add,1);add+=__shfl_xor_sync(0xffffffff,add,2);
   sum=sum*a+add;
   unsigned pr[4];
   #pragma unroll
   for(int i=0;i<4;i++){half2 packed=__floats2half2_rn(score.x[2*i],score.x[2*i+1]);pr[i]=*(unsigned*)&packed;}
   #pragma unroll
   for(int n=0;n<4;n++) {
     #pragma unroll
     for(int i=0;i<8;i++)acc[n].x[i]*=a;
     #pragma unroll
     for(int j=0;j<2;j++) {
       unsigned addr=__cvta_generic_to_shared(kv[warp][1]+(lane%16)*72+n*16+j*8),v0,v1;
       asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];":"=r"(v0),"=r"(v1):"r"(addr));
       float* z=acc[n].x+j*4;
       asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
         : "+f"(z[0]),"+f"(z[1]),"+f"(z[2]),"+f"(z[3]):"r"(pr[0]),"r"(pr[1]),"r"(pr[2]),"r"(pr[3]),"r"(v0),"r"(v1));
     }
   }
   __syncwarp();
 }
 long part=((long)(b*HKV+h)*total+ws)*8;
 #pragma unroll
 for(int n=0;n<4;n++) {
   #pragma unroll
   for(int i=0;i<8;i++) if(!(i&2)) {
     int row=lane/4,col=n*16+(lane%4)*2+(i&1)+(i&4)*2;
     pa[(part+row)*64+col]=acc[n].x[i];
   }
 }
 if(lane%4==0){pm[(part+lane/4)*2]=mx;pm[(part+lane/4)*2+1]=sum;}
 if(!FUSED)return;
 // All threads publish their stores before the CTA's arrival. The last CTA acquires the
 // release sequence, then makes that visibility available to the whole CTA via the barrier.
 __threadfence();__syncthreads();
 if(tid==0) {
   cuda::atomic_ref<int,cuda::thread_scope_device> count(counts[b*HKV+h]);
   last=(count.fetch_add(1,cuda::memory_order_acq_rel)==S-1);
 }
 __syncthreads();
 if(!last)return;
 for(int row=warp;row<8;row+=W) {
   long p0=(long)(b*HKV+h)*total*8+row;
   float m=-1e30f;
   for(int s=0;s<total;s++)m=fmaxf(m,pm[(p0+s*8)*2]);
   float den=0,x=0,y=0;
   for(int s=0;s<total;s++) {
     long p=p0+s*8;float w=exp2f(pm[p*2]-m);
     den+=w*pm[p*2+1];x+=w*pa[p*64+lane];y+=w*pa[p*64+lane+32];
   }
   x=x/den*vscale;y=y/den*vscale;
   float r=rsqrtf(warp_sum(x*x+y*y)/64.f+1e-5f);
   long col=(h*8+row)*64+lane;
   const __nv_bfloat16* gate=qkv+b*width+(HQ+2*HKV)*64+col;
   out[(long)b*HQ*64+col]=__float2bfloat16(x*r/(1.f+expf(-__bfloat162float(gate[0]))));
   out[(long)b*HQ*64+col+32]=__float2bfloat16(y*r/(1.f+expf(-__bfloat162float(gate[32]))));
 }
 __syncthreads();
 if(tid==0)counts[b*HKV+h]=0;
}
void attention(at::Tensor qkv,at::Tensor cs,at::Tensor kc,at::Tensor vc,at::Tensor pt,at::Tensor seq,at::Tensor loc,
 at::Tensor ks,at::Tensor vs,at::Tensor pa,at::Tensor pm,at::Tensor count,at::Tensor out,
 int splits,int page,double scale,int stages,int warps,bool fused) {
 int hkv=kc.size(2),hq=(qkv.size(2)/64-2*hkv)/2;
 auto stream=c10::cuda::getCurrentCUDAStream();dim3 grid(splits,qkv.size(0),hkv);
 #define LAUNCH(S,W,P,F) sliding<S,W,P,F><<<grid,W*32,0,stream>>>((__nv_bfloat16*)qkv.data_ptr(),cs.data_ptr<float>(),(unsigned char*)kc.data_ptr(),(unsigned char*)vc.data_ptr(),pt.data_ptr<int>(),seq.data_ptr<int>(),loc.data_ptr<long>(),ks.data_ptr<float>(),vs.data_ptr<float>(),pa.data_ptr<float>(),pm.data_ptr<float>(),count.data_ptr<int>(),(__nv_bfloat16*)out.data_ptr(),hq,hkv,pt.size(1),splits,scale)
 #define FF(S,W,P) if(fused){LAUNCH(S,W,P,true);}else{LAUNCH(S,W,P,false);}
 #define PP(S,W) if(page==1){FF(S,W,1);}else{FF(S,W,64);}
 #define WW(S) if(warps==1){PP(S,1);}else if(warps==2){PP(S,2);}else{PP(S,4);}
 if(stages==2){WW(2);}else{WW(3);}
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''
_build=Path(__file__).resolve().parent/'.build';_build.mkdir(exist_ok=True)
module=load_inline(name='codex_sliding_exp',build_directory=str(_build),cpp_sources=CPP,cuda_sources=CUDA,functions=['attention'],extra_cuda_cflags=['-gencode=arch=compute_90a,code=sm_90a','-O3','--use_fast_math','-lineinfo','--ptxas-options=-v'],extra_cflags=['-O3'],verbose=True)

def run(data,splits=4,warps=4,stages=3,fused=True):
    import submission,triton
    c=data['config'];b=c['batch'];h=c['num_kv_heads'];g=c['num_q_heads']//h
    assert c['q_len']==1 and g==8 and c['head_dim']==64 and c['window']==2048 and c['page_size'] in (1,64)
    qkv=data['qkv'];dev=qkv.device;s=splits*warps
    pa=torch.empty((b,h,s,8,64),dtype=torch.float32,device=dev)
    pm=torch.empty((b,h,s,8,2),dtype=torch.float32,device=dev)
    out=torch.empty((b,1,c['num_q_heads']*64),dtype=torch.bfloat16,device=dev)
    count=submission._counters(b*h,dev)
    module.attention(qkv,submission._cos_sin(c['rope_theta'],64,dev),data['k_cache'],data['v_cache'],data['page_table'],data['seq_lens'],data['cache_loc'],data['k_scale'],data['v_scale'],pa,pm,count,out,splits,c['page_size'],c['sm_scale']*c['qk_scale_factor']*submission.LOG2E,stages,warps,fused)
    if not fused:
        submission._combine[(b,c['num_q_heads'])](qkv,pa,pm,data['v_scale'],out,HQ=c['num_q_heads'],HKV=h,G=g,L=1,M=8,D=64,S=s,S_PAD=triton.next_power_of_2(s),SB=min(128,triton.next_power_of_2(s)),PDL=False,num_warps=4 if s>32 else 2)
    return out

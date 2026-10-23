"""Diagnostic synchronous-gather version of the sliding WGMMA cluster prototype.

Returns (output, partial_accumulators, partial_max_and_sum). The masked descriptor
is correct on the diagnostic cases. This is not a production candidate.
"""
import torch
from pathlib import Path
from torch.utils.cpp_extension import load_inline

CPP = r'''
#include <torch/extension.h>
void attention(torch::Tensor qkv, torch::Tensor cs, torch::Tensor kc, torch::Tensor vc, torch::Tensor pt,
 torch::Tensor seq, torch::Tensor loc, torch::Tensor ks, torch::Tensor vs,
 torch::Tensor pa, torch::Tensor pm, torch::Tensor out, int splits, int page, double scale, int stages);
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
#include <cooperative_groups.h>
#include <cuda/atomic>
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
 cuda::atomic_ref<unsigned,cuda::thread_scope_block> flag(*p); return flag.load(cuda::memory_order_acquire);
}
__device__ __forceinline__ void release(unsigned* p,unsigned v) {
 cuda::atomic_ref<unsigned,cuda::thread_scope_block> flag(*p); flag.store(v,cuda::memory_order_release);
}

template<int STAGES, int PAGE, int SP>
__global__ __launch_bounds__(160,4) void verify(const __nv_bfloat16* __restrict__ qkv, const float* cs, unsigned char* kc, unsigned char* vc,
 const int* __restrict__ pt, const int* __restrict__ seqs, const long* __restrict__ loc,
 const float* ks, const float* vs, float* __restrict__ pa, float* __restrict__ pm, __nv_bfloat16* out,
 int HQ,int HKV,int MP,int SPLITS,float SCALE) {
 constexpr int N=32, M=64, D=64;
 __shared__ __align__(16) struct Scratch {
   unsigned char raw[STAGES][2][N*(D+16)];
   struct {float a[64*64];float ml[64*2];} parts;
 } scratch;
 auto& raw=scratch.raw;
 auto cluster=cooperative_groups::this_cluster();
 cluster.sync();
 __shared__ __align__(16) unsigned char own[2][8*D];
 __shared__ __align__(32) half V[N*(D+8)];
 __shared__ __align__(16) unsigned char Q[M*(D+16)];
 __shared__ unsigned ready[STAGES];
 const unsigned tid=threadIdx.x, warp=tid/32, lane=tid%32;
 const int b=blockIdx.y, h=blockIdx.z, split=blockIdx.x;
 const int seq=seqs[b], width=(2*HQ+2*HKV)*D, G=8;
 const int base=max(0,seq-2047);
 const int chunk=((seq-base+N*SP-1)/(N*SP))*N;
 const int lo=base+split*chunk, hi=min(seq,lo+chunk);
 const int tiles=max(0,(hi-lo+N-1)/N)+(split==0);
 const float kscale=*ks, vscale=*vs;
 for(int i=tid;i<STAGES;i+=blockDim.x) ready[i]=0;
 // Each warp normalizes one row at a time, exactly the same query FP8 rounding as FA3.
 for(int row=warp;row<M;row+=5) {
   int tok=b*8+row/G, head=h*G+row%G;
   float x=__bfloat162float(qkv[tok*width+head*D+lane]);
   float y=__bfloat162float(qkv[tok*width+head*D+lane+32]);
   float r=rsqrtf(warp_sum(x*x+y*y)/64.f+1e-5f);
   float xp=__bfloat162float(qkv[tok*width+head*D+(lane^1)]);
   float yp=__bfloat162float(qkv[tok*width+head*D+32+(lane^1)]);
   int pos=seq+row/G; float sign=(lane&1)?1.f:-1.f;
   x=x*cs[(long)pos*64+lane/2]+sign*xp*cs[(long)pos*64+32+lane/2];
   y=y*cs[(long)pos*64+16+lane/2]+sign*yp*cs[(long)pos*64+48+lane/2];
   Q[row*(D+16)+lane]=to8(x*r); Q[row*(D+16)+lane+32]=to8(y*r);
 }
 // New keys are handled without depending on concurrent writes to the shared cache.
 if(split==0) for(int row=warp;row<8;row+=5) {
   int tok=b*8+row;
   float x=__bfloat162float(qkv[tok*width+(HQ+h)*D+lane]);
   float y=__bfloat162float(qkv[tok*width+(HQ+h)*D+lane+32]);
   float r=rsqrtf(warp_sum(x*x+y*y)/64.f+1e-5f)/kscale;
   float xp=__bfloat162float(qkv[tok*width+(HQ+h)*D+(lane^1)]);
   float yp=__bfloat162float(qkv[tok*width+(HQ+h)*D+32+(lane^1)]);
   int pos=seq+row;float sign=(lane&1)?1.f:-1.f;
   x=x*cs[(long)pos*64+lane/2]+sign*xp*cs[(long)pos*64+32+lane/2];
   y=y*cs[(long)pos*64+16+lane/2]+sign*yp*cs[(long)pos*64+48+lane/2];
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
 } else {
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
     q[d][i]=*(unsigned*)(Q+row*(D+16)+col);
   }
 }
 for(int tile=0;tile<tiles;tile++) {
   int stage=tile%STAGES;
   bool newtile=split==0 && tile==tiles-1;
   if(!newtile) {
     for(unsigned i=tid;i<N*4;i+=128) {
       int t=min(lo+tile*N+int(i)/4,hi-1);
       long row=((long)pt[b*MP+t/PAGE]*PAGE+t%PAGE)*HKV*D+h*D;
       copy16(raw[stage][0]+(i/32)*512+(i%4)*128+((i/4)%8)*16,kc+row+(i%4)*16);
       copy16(raw[stage][1]+(i/4)*(D+16)+(i%4)*16,vc+row+(i%4)*16);
     }
   } else {
     for(unsigned i=tid;i<N*D;i+=128) {
       raw[stage][0][(i/(D*8))*512+((i%D)/16)*128+((i/D)%8)*16+i%16]=i<8*D?own[0][i]:0;
       raw[stage][1][(i/D)*(D+16)+i%D]=i<8*D?own[1][i]:0;
     }
   }
   asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;":::"memory");
   consumer_barrier();
   asm volatile("fence.proxy.async.shared::cta;":::"memory");
   #pragma unroll
   for(unsigned j=0;j<N*D/256;j++) {
     unsigned i=2*tid+j*256;
     unsigned target=(i/D)*(D+8)+i%D;
     __half2_raw vv=__nv_cvt_fp8x2_to_halfraw2(*(unsigned short*)(raw[stage][1]+(i/D)*(D+16)+i%D),__NV_E4M3);
     *(half2*)(V+target)=vv;
   }
   consumer_barrier();
   wmma::fragment<wmma::accumulator,16,16,16,float> score[2];
   wmma::fill_fragment(score[0],0.f); wmma::fill_fragment(score[1],0.f);
   asm volatile("" : "+f"(score[0].x[0]), "+f"(score[0].x[1]), "+f"(score[0].x[2]), "+f"(score[0].x[3]), "+f"(score[0].x[4]), "+f"(score[0].x[5]), "+f"(score[0].x[6]), "+f"(score[0].x[7]), "+f"(score[1].x[0]), "+f"(score[1].x[1]), "+f"(score[1].x[2]), "+f"(score[1].x[3]), "+f"(score[1].x[4]), "+f"(score[1].x[5]), "+f"(score[1].x[6]), "+f"(score[1].x[7]) :: "memory");
   asm volatile("wgmma.fence.sync.aligned;":::"memory");
   #pragma unroll
   for(int d=0;d<2;d++) {
     unsigned long desc=(((unsigned long)__cvta_generic_to_shared(raw[stage][0]+d*256)&0x3ffffull)>>4)
                        | (8ull<<16) | (32ull<<32);
     asm volatile("wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, {%16,%17,%18,%19}, %20, 1, 1, 1;"
       : "+f"(score[0].x[0]), "+f"(score[0].x[1]), "+f"(score[0].x[2]), "+f"(score[0].x[3]), "+f"(score[0].x[4]), "+f"(score[0].x[5]), "+f"(score[0].x[6]), "+f"(score[0].x[7]), "+f"(score[1].x[0]), "+f"(score[1].x[1]), "+f"(score[1].x[2]), "+f"(score[1].x[3]), "+f"(score[1].x[4]), "+f"(score[1].x[5]), "+f"(score[1].x[6]), "+f"(score[1].x[7])
       : "r"(q[d][0]),"r"(q[d][1]),"r"(q[d][2]),"r"(q[d][3]),"l"(desc));
   }
   asm volatile("wgmma.commit_group.sync.aligned;\nwgmma.wait_group.sync.aligned 0;":::"memory");
   asm volatile("" : "+f"(score[0].x[0]), "+f"(score[0].x[1]), "+f"(score[0].x[2]), "+f"(score[0].x[3]), "+f"(score[0].x[4]), "+f"(score[0].x[5]), "+f"(score[0].x[6]), "+f"(score[0].x[7]), "+f"(score[1].x[0]), "+f"(score[1].x[1]), "+f"(score[1].x[2]), "+f"(score[1].x[3]), "+f"(score[1].x[4]), "+f"(score[1].x[5]), "+f"(score[1].x[6]), "+f"(score[1].x[7]) :: "memory");
   bool isnew=split==0 && tile==tiles-1;
   // Mask only the final cached tile and the new-token tile.
   if(isnew || tile==0 || lo+(tile+1)*N>hi) {
     int bounds[2]={isnew?int(warp)*2:hi-lo-tile*N-1,isnew?int(warp)*2+1:hi-lo-tile*N-1};
     #pragma unroll
     for(int n=0;n<2;n++) {
       #pragma unroll
       for(int i=0;i<8;i++) {
         int r=(i&2)/2;
         int col=n*16+(lane%4)*2+(i&1)+(i&4)*2;
         if(col>bounds[r] || (!isnew && lo+tile*N+col<seq+int(warp)*2+r-2047)) score[n].x[i]=-1e30f;
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
         unsigned addr=__cvta_generic_to_shared(V+(d*16+lane%16)*(D+8)+n*16+j*8);
         unsigned v0,v1;
         asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];":"=r"(v0),"=r"(v1):"r"(addr));
         float* z=acc[n].x+j*4;
         asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
           : "+f"(z[0]),"+f"(z[1]),"+f"(z[2]),"+f"(z[3])
           : "r"(pr[d][0]),"r"(pr[d][1]),"r"(pr[d][2]),"r"(pr[d][3]),"r"(v0),"r"(v1));
       }
     }
   }
   consumer_barrier();

 }
 consumer_barrier();
 #pragma unroll
 for(int n=0;n<4;n++) {
 wmma::store_matrix_sync(scratch.parts.a+warp*16*D+n*16,acc[n],D,wmma::mem_row_major);
 long p=((long)(b*HKV+h)*SP+split)*64+warp*16;
 wmma::store_matrix_sync(pa+p*D+n*16,acc[n],D,wmma::mem_row_major);
 }
 if(lane%4==0) for(int r=0;r<2;r++) {
   int row=warp*16+lane/4+r*8;
   scratch.parts.ml[row*2]=mx[r];scratch.parts.ml[row*2+1]=sum[r];
 long p=((long)(b*HKV+h)*SP+split)*64+row;pm[p*2]=mx[r];pm[p*2+1]=sum[r];
 }
 } // consumers
 __syncthreads();
 cluster.sync();
 for(int row=split*5+warp;row<64;row+=SP*5) {
   float mm[SP],ll[SP],xx[SP],yy[SP];float m=-1e30f;
   #pragma unroll
   for(int s=0;s<SP;s++) {
     const float* a=cluster.map_shared_rank(scratch.parts.a,s);
     const float* ml=cluster.map_shared_rank(scratch.parts.ml,s);
     mm[s]=ml[row*2];ll[s]=ml[row*2+1];xx[s]=a[row*64+lane];yy[s]=a[row*64+lane+32];
     m=fmaxf(m,mm[s]);
   }
   float den=0,x=0,y=0;
   #pragma unroll
   for(int s=0;s<SP;s++){float e=exp2f(mm[s]-m);den+=e*ll[s];x+=e*xx[s];y+=e*yy[s];}
   x=x/den*vscale;y=y/den*vscale;float r=rsqrtf(warp_sum(x*x+y*y)/64.f+1e-5f);
   long tok=b*8+row/8,col=(h*8+row%8)*64+lane;
   const __nv_bfloat16* gate=qkv+tok*width+(HQ+2*HKV)*64+col;
   out[tok*HQ*64+col]=__float2bfloat16(x*r/(1.f+expf(-__bfloat162float(gate[0]))));
   out[tok*HQ*64+col+32]=__float2bfloat16(y*r/(1.f+expf(-__bfloat162float(gate[32]))));
 }
 cluster.sync();
}

void attention(at::Tensor qkv, at::Tensor cs, at::Tensor kc, at::Tensor vc, at::Tensor pt,
 at::Tensor seq, at::Tensor loc, at::Tensor ks, at::Tensor vs,
 at::Tensor pa, at::Tensor pm, at::Tensor out, int splits, int page, double scale, int stages) {
 int hkv=kc.size(2), hq=(qkv.size(2)/64-2*hkv)/2;
 auto stream=c10::cuda::getCurrentCUDAStream();
 dim3 grid(splits,qkv.size(0),hkv);
 cudaLaunchConfig_t config={};config.gridDim=grid;config.blockDim=dim3(160);config.stream=stream;
 cudaLaunchAttribute attribute={};attribute.id=cudaLaunchAttributeClusterDimension;
 attribute.val.clusterDim.x=splits;attribute.val.clusterDim.y=1;attribute.val.clusterDim.z=1;
 config.attrs=&attribute;config.numAttrs=1;
 #define LAUNCH(ST,P,S) C10_CUDA_CHECK(cudaLaunchKernelEx(&config,verify<ST,P,S>,(__nv_bfloat16*)qkv.data_ptr(),cs.data_ptr<float>(),(unsigned char*)kc.data_ptr(),(unsigned char*)vc.data_ptr(),pt.data_ptr<int>(),seq.data_ptr<int>(),loc.data_ptr<long>(),ks.data_ptr<float>(),vs.data_ptr<float>(),pa.data_ptr<float>(),pm.data_ptr<float>(),(__nv_bfloat16*)out.data_ptr(),hq,hkv,int(pt.size(1)),splits,float(scale)))
 #define SS(ST,P) if(splits==1){LAUNCH(ST,P,1);}else if(splits==2){LAUNCH(ST,P,2);}else if(splits==4){LAUNCH(ST,P,4);}else{LAUNCH(ST,P,8);}
 #define PP(ST) if(page==1){SS(ST,1);}else{SS(ST,64);}
 if(stages==2){PP(2);}else{PP(3);}
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''
_build = Path(__file__).resolve().parent / '.build_verify_coop'
_build.mkdir(exist_ok=True)
module = load_inline(name='codex_sliding_verify_coop', build_directory=str(_build), cpp_sources=CPP, cuda_sources=CUDA, functions=['attention'],
                    extra_cuda_cflags=['-gencode=arch=compute_90a,code=sm_90a','-O3','--use_fast_math','-lineinfo','--ptxas-options=-v'],
                    extra_cflags=['-O3'], verbose=True)

def run(data, splits=None, stages=3):
    import submission, triton
    c=data['config']; b=c['batch']; h=c['num_kv_heads']; g=c['num_q_heads']//h
    assert c['q_len']==8 and g==8 and c['head_dim']==64 and c['window']==2048 and c['page_size'] in (1,64)
    s=splits or 4
    assert s in (1,2,4,8) and stages in (2,3)
    qkv=data['qkv']; device=qkv.device
    pa=torch.empty((b,h,s,64,64),dtype=torch.float32,device=device)
    pm=torch.empty((b,h,s,64,2),dtype=torch.float32,device=device)
    out=torch.empty((b,8,c['num_q_heads']*64),dtype=torch.bfloat16,device=device)
    module.attention(qkv,submission._cos_sin(c['rope_theta'],64,device),data['k_cache'],data['v_cache'],data['page_table'],data['seq_lens'],data['cache_loc'],
                     data['k_scale'],data['v_scale'],pa,pm,out,s,c['page_size'],c['sm_scale']*c['qk_scale_factor']*submission.LOG2E,stages)
    return out,pa,pm

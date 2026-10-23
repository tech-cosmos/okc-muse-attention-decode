"""Experimental native multi-warp FP32 split reduction; no attention plan changes."""
from torch.utils.cpp_extension import load_inline

CUDA = r'''
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
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
 constexpr int NS=N/32;
 float2 ml[NS];
 float m=-1e30f;
 #pragma unroll
 for(int i=0;i<NS;i++) {
   int s=warp*N+i*32+lane;
   ml[i]=s<S?read2((const float*)(pm+base+s*MG)):make_float2(-1e30f,0.f);
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
void reduce(at::Tensor qkv,at::Tensor pa,at::Tensor pm,at::Tensor vs,at::Tensor out,int hq,int hkv,int length,int splits,int warps,bool pdl) {
 auto stream=c10::cuda::getCurrentCUDAStream();int b=qkv.size(0);
 cudaLaunchConfig_t config={};config.gridDim=dim3(b*length*hq);config.blockDim=dim3(warps*32);config.stream=stream;
 cudaLaunchAttribute attr={};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;attr.val.programmaticStreamSerializationAllowed=pdl;
 config.attrs=&attr;config.numAttrs=1;
 #define RUN(W,N,P) C10_CUDA_CHECK(cudaLaunchKernelEx(&config,reduce_kernel<W,N,P>,(const __nv_bfloat16*)qkv.data_ptr(),pa.data_ptr<float>(),(const float2*)pm.data_ptr<float>(),vs.data_ptr<float>(),(__nv_bfloat16*)out.data_ptr(),hq,hkv,length,b,splits))
 #define PP(W,N) if(pdl){RUN(W,N,true);}else{RUN(W,N,false);}
 #define NW(W) if(splits<=W*32){PP(W,32);}else if(splits<=W*64){PP(W,64);}else {PP(W,128);}
 if(warps==1){NW(1);}else if(warps==2){NW(2);}else if(warps==4){NW(4);}else{NW(8);}
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''
MOD = load_inline(name='muse_reduce_exp', cpp_sources='''
#include <torch/extension.h>
void reduce(torch::Tensor qkv,torch::Tensor pa,torch::Tensor pm,torch::Tensor vs,torch::Tensor out,int hq,int hkv,int length,int splits,int warps,bool pdl);
''', cuda_sources=CUDA, functions=['reduce'], extra_cuda_cflags=['-gencode=arch=compute_90a,code=sm_90a','-O3','--use_fast_math','-Xptxas=-v'], extra_cflags=['-O3'], verbose=False)
class Wrapper:
    def __init__(self,warps): self.warps=warps
    def __getitem__(self,grid):
        def run(qkv,pa,pm,vs,out,**kw):
            MOD.reduce(qkv,pa,pm,vs,out,kw['HQ'],kw['HKV'],kw['L'],kw['S'],self.warps,kw['PDL'])
        return run

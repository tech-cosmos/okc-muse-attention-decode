"""Native output reduction experiment, paired with the unmodified Triton attention."""
import torch
from pathlib import Path
from torch.utils.cpp_extension import load_inline
CPP=r'''
#include <torch/extension.h>
void combine(torch::Tensor qkv,torch::Tensor pa,torch::Tensor pm,torch::Tensor vs,torch::Tensor out,int hq,int hkv,int length,int splits,int warps,bool pdl);
'''
CUDA=r'''
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
template<int S,int W,bool PDL>
__global__ __launch_bounds__(W*32) void combine_kernel(const __nv_bfloat16* __restrict__ qkv,
 const float* pa,const float2* pm,const float* vs,__nv_bfloat16* out,
 int HQ,int HKV,int L,int B) {
 unsigned lane=threadIdx.x%32,warp=threadIdx.x/32;
 int row=blockIdx.x*W+warp;
 if(row>=B*L*HQ)return;
 int tok=row/HQ,head=row%HQ,b=tok/L,g=HQ/HKV,h=head/g;
 int width=(2*HQ+2*HKV)*64,rr=(tok%L)*g+head%g,MG=L*g;
 int base=((b*HKV+h)*S)*MG+rr;
 __nv_bfloat162 gate2=*(const __nv_bfloat162*)(qkv+tok*width+(HQ+2*HKV)*64+head*64+2*lane);
 float2 gate=__bfloat1622float2(gate2);
 float vscale=*vs;
 if(PDL)asm volatile("griddepcontrol.wait;":::"memory");
 float2 ml[S];float2 a[S];float m=-1e30f;
 #pragma unroll
 for(int s=0;s<S;s++) {
   const float* mp=(const float*)(pm+base+s*MG);
   const float* ap=pa+(base+s*MG)*64+2*lane;
   asm volatile("ld.global.cg.v2.f32 {%0,%1}, [%2];" : "=f"(ml[s].x), "=f"(ml[s].y) : "l"(mp) : "memory");
   asm volatile("ld.global.cg.v2.f32 {%0,%1}, [%2];" : "=f"(a[s].x), "=f"(a[s].y) : "l"(ap) : "memory");
   m=fmaxf(m,ml[s].x);
 }
 float den=0,x=0,y=0;
 #pragma unroll
 for(int s=0;s<S;s++) {
   float w=exp2f(ml[s].x-m);
   den=fmaf(w,ml[s].y,den);x=fmaf(w,a[s].x,x);y=fmaf(w,a[s].y,y);
 }
 x=x/den*vscale;y=y/den*vscale;
 float sq=x*x+y*y;
 #pragma unroll
 for(int d=16;d;d/=2)sq+=__shfl_xor_sync(0xffffffff,sq,d);
 float r=rsqrtf(sq/64.f+1e-5f);
 x=x*r/(1.f+expf(-gate.x));y=y*r/(1.f+expf(-gate.y));
 *(__nv_bfloat162*)(out+row*64+2*lane)=__floats2bfloat162_rn(x,y);
}
void combine(at::Tensor qkv,at::Tensor pa,at::Tensor pm,at::Tensor vs,at::Tensor out,int hq,int hkv,int length,int splits,int warps,bool pdl) {
 auto stream=c10::cuda::getCurrentCUDAStream();int b=qkv.size(0);
 cudaLaunchConfig_t config={};config.gridDim=dim3((b*length*hq+warps-1)/warps);config.blockDim=dim3(warps*32);config.stream=stream;
 cudaLaunchAttribute attr={};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;attr.val.programmaticStreamSerializationAllowed=pdl;
 config.attrs=&attr;config.numAttrs=1;
 #define RUN(S,W,P) C10_CUDA_CHECK(cudaLaunchKernelEx(&config,combine_kernel<S,W,P>,(const __nv_bfloat16*)qkv.data_ptr(),pa.data_ptr<float>(),(const float2*)pm.data_ptr<float>(),vs.data_ptr<float>(),(__nv_bfloat16*)out.data_ptr(),hq,hkv,length,b))
 #define PP(S,W) if(pdl){RUN(S,W,true);}else{RUN(S,W,false);}
 #define WW(S) if(warps==1){PP(S,1);}else if(warps==2){PP(S,2);}else if(warps==4){PP(S,4);}else{PP(S,8);}
 switch(splits){
 case 4:{WW(4);break;}case 8:{WW(8);break;}case 12:{WW(12);break;}case 16:{WW(16);break;}case 24:{WW(24);break;}case 32:{WW(32);break;}case 64:{WW(64);break;}case 128:{WW(128);break;}
 default:TORCH_CHECK(false,"unsupported splits");}
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''
_build=Path(__file__).resolve().parent/'.build_merge';_build.mkdir(exist_ok=True)
module=load_inline(name='codex_merge_exp',build_directory=str(_build),cpp_sources=CPP,cuda_sources=CUDA,functions=['combine'],extra_cuda_cflags=['-gencode=arch=compute_90a,code=sm_90a','-O3','--use_fast_math','-lineinfo','--ptxas-options=-v'],extra_cflags=['-O3'],verbose=True)
class Merge:
    def __init__(self,original,warps=4):self.original=original;self.warps=warps
    def __getitem__(self,grid):
        def run(qkv,pa,pm,vs,out,**kw):
            if kw['S'] not in (4,8,12,16,24,32,64,128):return self.original[grid](qkv,pa,pm,vs,out,**kw)
            module.combine(qkv,pa,pm,vs,out,kw['HQ'],kw['HKV'],kw['L'],kw['S'],self.warps,kw['PDL'])
        return run

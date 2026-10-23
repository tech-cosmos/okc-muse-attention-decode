

#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
template<int S,int W,bool PDL,int HQ,int L>
__global__ __launch_bounds__(W*32) void combine_kernel(const __nv_bfloat16* __restrict__ qkv,
 const float* pa,const float2* pm,const float* vs,__nv_bfloat16* out,
 int B) {
 constexpr int HKV=HQ/8;
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
   // These are produced by the concurrently launched predecessor. Force coherent loads
   // after the dependency wait; read-only loads may be hoisted across inline assembly.
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
void combine(at::Tensor qkv,at::Tensor pa,at::Tensor pm,at::Tensor vs,at::Tensor out,int hq,int hkv,int length,int splits,bool pdl) {
 auto stream=c10::cuda::getCurrentCUDAStream();int b=qkv.size(0);constexpr int warps=4;
 cudaLaunchConfig_t config={};config.gridDim=dim3((b*length*hq+warps-1)/warps);config.blockDim=dim3(warps*32);config.stream=stream;
 cudaLaunchAttribute attr={};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;attr.val.programmaticStreamSerializationAllowed=pdl;
 config.attrs=&attr;config.numAttrs=1;
 #define RUN(S,W,P,H,L) C10_CUDA_CHECK(cudaLaunchKernelEx(&config,combine_kernel<S,W,P,H,L>,(const __nv_bfloat16*)qkv.data_ptr(),pa.data_ptr<float>(),(const float2*)pm.data_ptr<float>(),vs.data_ptr<float>(),(__nv_bfloat16*)out.data_ptr(),b))
 #define LL(S,W,P,H) if(length==1){RUN(S,W,P,H,1);}else{RUN(S,W,P,H,8);}
 #define HH(S,W,P) if(hq==16){LL(S,W,P,16);}else{LL(S,W,P,128);}
 #define PP(S,W) if(pdl){HH(S,W,true);}else{HH(S,W,false);}
 #define WW(S) PP(S,4)
 switch(splits){
 case 4:{WW(4);break;}case 8:{WW(8);break;}case 16:{WW(16);break;}case 32:{WW(32);break;}
 default:TORCH_CHECK(false,"unsupported splits");}
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}

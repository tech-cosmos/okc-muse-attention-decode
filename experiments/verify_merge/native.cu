#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
__device__ __forceinline__ float2 read2(const float* p) {
 float2 v;asm volatile("ld.global.cg.v2.f32 {%0,%1}, [%2];":"=f"(v.x),"=f"(v.y):"l"(p):"memory");return v;
}
__device__ __forceinline__ float4 read4(const float* p) {
 float4 v;asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];":"=f"(v.x),"=f"(v.y),"=f"(v.z),"=f"(v.w):"l"(p):"memory");return v;
}
__device__ __forceinline__ float sum32(float x) {
 #pragma unroll
 for(int d=16;d;d/=2)x+=__shfl_xor_sync(0xffffffff,x,d);return x;
}
__device__ __forceinline__ float max32(float x) {
 #pragma unroll
 for(int d=16;d;d/=2)x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,d));return x;
}
// Four independent output rows per CTA. Striped=true maps four split groups
// across each warp and eight adjacent dimension groups per split.
template<int CAP,bool STRIPED>
__global__ __launch_bounds__(128) void merge_rows(const __nv_bfloat16* qkv,const float* pa,const float* pm,
 const float* vs,__nv_bfloat16* out,int HQ,int HKV,int L,int S,bool pdl) {
 unsigned lane=threadIdx.x%32,warp=threadIdx.x/32;
 int row=blockIdx.x*4+warp;
 // Physical partial order [b,hkv,split,token*G+head_in_group,64].
 int G=HQ/HKV,MG=L*G,bh=row/MG,rr=row%MG;
 int b=bh/HKV,h=bh%HKV,t=rr/G,head=h*G+rr%G,tok=b*L+t;
 int base=bh*S*MG+rr,width=(2*HQ+2*HKV)*64;
 const __nv_bfloat16* gatep=qkv+tok*width+(HQ+2*HKV)*64+head*64;
 float vscale=*vs;
 if constexpr(!STRIPED) {
  float2 gate=__bfloat1622float2(*(const __nv_bfloat162*)(gatep+lane*2));
  if(pdl)asm volatile("griddepcontrol.wait;":::"memory");
  constexpr int NS=(CAP+31)/32;
  float2 ml[NS];float m=-1e30f;
  #pragma unroll
  for(int i=0;i<NS;i++){int s=i*32+lane;ml[i]=s<S?read2(pm+(base+s*MG)*2):make_float2(-1e30f,0);m=fmaxf(m,ml[i].x);}
  m=max32(m);float w[NS],den=0;
  #pragma unroll
  for(int i=0;i<NS;i++){w[i]=exp2f(ml[i].x-m);den=fmaf(w[i],ml[i].y,den);}
  den=sum32(den);float2 a[CAP];
  #pragma unroll
  for(int s=0;s<CAP;s++)a[s]=s<S?read2(pa+(base+s*MG)*64+2*lane):make_float2(0,0);
  float x=0,y=0;
  #pragma unroll
  for(int s=0;s<CAP;s++){float ws=__shfl_sync(0xffffffff,w[s/32],s%32);x=fmaf(ws,a[s].x,x);y=fmaf(ws,a[s].y,y);}
  x=x/den*vscale;y=y/den*vscale;float r=rsqrtf(sum32(x*x+y*y)/64.f+1e-5f);
  x=x*r/(1+expf(-gate.x));y=y*r/(1+expf(-gate.y));
  *(__nv_bfloat162*)(out+(tok*HQ+head)*64+lane*2)=__floats2bfloat162_rn(x,y);
 } else {
  constexpr int NS=(CAP+3)/4;
  unsigned sg=lane/8,dc=lane%8;float gate[8];
  #pragma unroll
  for(int j=0;j<4;j++){float2 v=__bfloat1622float2(*(const __nv_bfloat162*)(gatep+dc*8+j*2));gate[2*j]=v.x;gate[2*j+1]=v.y;}
  if(pdl)asm volatile("griddepcontrol.wait;":::"memory");
  float2 ml[NS];float m=-1e30f;
  #pragma unroll
  for(int i=0;i<NS;i++){int s=i*4+sg;ml[i]=s<S?read2(pm+(base+s*MG)*2):make_float2(-1e30f,0);m=fmaxf(m,ml[i].x);}
  m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,8));m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,16));
  float4 av[NS][2];
  #pragma unroll
  for(int i=0;i<NS;i++){
   int s=i*4+sg;
   #pragma unroll
   for(int j=0;j<2;j++)av[i][j]=s<S?read4(pa+(base+s*MG)*64+dc*8+j*4):make_float4(0,0,0,0);
  }
  float acc[8]={},den=0;
  #pragma unroll
  for(int i=0;i<NS;i++){
   float w=exp2f(ml[i].x-m);den=fmaf(w,ml[i].y,den);
   #pragma unroll
   for(int j=0;j<2;j++){
    acc[4*j]=fmaf(w,av[i][j].x,acc[4*j]);acc[4*j+1]=fmaf(w,av[i][j].y,acc[4*j+1]);
    acc[4*j+2]=fmaf(w,av[i][j].z,acc[4*j+2]);acc[4*j+3]=fmaf(w,av[i][j].w,acc[4*j+3]);
   }
  }
  den+=__shfl_xor_sync(0xffffffff,den,8);den+=__shfl_xor_sync(0xffffffff,den,16);
  float sq=0;
  #pragma unroll
  for(int j=0;j<8;j++){
   float x=acc[j];x+=__shfl_xor_sync(0xffffffff,x,8);x+=__shfl_xor_sync(0xffffffff,x,16);
   acc[j]=x/den*vscale;sq=fmaf(acc[j],acc[j],sq);
  }
  #pragma unroll
  for(int d=4;d;d/=2)sq+=__shfl_xor_sync(0xffffffff,sq,d);
  float r=rsqrtf(sq/64.f+1e-5f);
  if(sg==0){
   #pragma unroll
   for(int j=0;j<4;j++){
    float x=acc[2*j]*r/(1+expf(-gate[2*j])),y=acc[2*j+1]*r/(1+expf(-gate[2*j+1]));
    *(__nv_bfloat162*)(out+(tok*HQ+head)*64+dc*8+2*j)=__floats2bfloat162_rn(x,y);
   }
  }
 }
}
void merge(at::Tensor qkv,at::Tensor pa,at::Tensor pm,at::Tensor vs,at::Tensor out,int hq,int hkv,int length,int splits,bool striped,bool pdl) {
 TORCH_CHECK(pa.size(3)==length*(hq/hkv) && pa.size(4)==64 && splits<=64,"unsupported partial layout");
 auto stream=c10::cuda::getCurrentCUDAStream();int rows=qkv.size(0)*length*hq;
 TORCH_CHECK(rows%4==0,"requires complete groups of four rows");
 cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(rows/4);cfg.blockDim=dim3(128);cfg.stream=stream;
 cudaLaunchAttribute attr={};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;attr.val.programmaticStreamSerializationAllowed=pdl;cfg.attrs=&attr;cfg.numAttrs=1;
 #define RUN(C,T) C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg,merge_rows<C,T>,(const __nv_bfloat16*)qkv.data_ptr(),pa.data_ptr<float>(),pm.data_ptr<float>(),vs.data_ptr<float>(),(__nv_bfloat16*)out.data_ptr(),hq,hkv,length,splits,pdl))
 #define PICK(T) if(splits<=16){RUN(16,T);}else if(splits<=32){RUN(32,T);}else{RUN(64,T);}
 if(striped){PICK(true);}else{PICK(false);}
 #undef PICK
 #undef RUN
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}

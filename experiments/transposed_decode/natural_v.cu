#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
__device__ __forceinline__ float sum32(float v){
 #pragma unroll
 for(int d=16;d;d/=2)v+=__shfl_xor_sync(0xffffffff,v,d);
 return v;
}
__device__ __forceinline__ unsigned char to8(float x){return __nv_cvt_float_to_fp8(x,__NV_SATFINITE,__NV_E4M3);}
__device__ __forceinline__ unsigned sw64(unsigned i){return i^(((i>>7)&3)<<4);}
__device__ __forceinline__ unsigned sw128(unsigned i){return i^(((i>>6)&7)<<3);}
__device__ __forceinline__ void cp16(void* d,const void* s,bool valid){
 asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"((unsigned)__cvta_generic_to_shared(d)),"l"(s),"r"(valid?16:0):"memory");
}
__device__ __forceinline__ unsigned long desc64(const void* p){return (((unsigned long)__cvta_generic_to_shared(p)&0x3ffff)>>4)|(1ull<<16)|(32ull<<32)|(2ull<<62);}
__device__ __forceinline__ unsigned long desc128(const void* p){return (((unsigned long)__cvta_generic_to_shared(p)&0x3ffff)>>4)|(1ull<<16)|(64ull<<32)|(1ull<<62);}
__device__ __forceinline__ void pin(float* x){asm volatile("":"+f"(x[0]),"+f"(x[1]),"+f"(x[2]),"+f"(x[3])::"memory");}
__device__ __forceinline__ void qk(float* x,unsigned long a,unsigned long b){
 asm volatile("wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0,%1,%2,%3}, %4,%5,1,1,1;":"+f"(x[0]),"+f"(x[1]),"+f"(x[2]),"+f"(x[3]):"l"(a),"l"(b));
}
__device__ __forceinline__ void pv(float* x,unsigned long a,unsigned long b){
 asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0,%1,%2,%3}, %4,%5,1,1,1,1,0;":"+f"(x[0]),"+f"(x[1]),"+f"(x[2]),"+f"(x[3]):"l"(a),"l"(b));
}
struct Shared {

 __align__(512) unsigned char K[3][2][4096];
 __align__(16) unsigned char VR[3][2][4096];
 __align__(1024) half VT[2][4096];
 __align__(512) unsigned char Q[2][512];
 __align__(1024) half P[2][512];
 unsigned char own[2][128];
 float stats[2][4][8];
};
template<int PAGE> __global__ __launch_bounds__(128) void decode(
 const __nv_bfloat16* qkv,unsigned char* kc,unsigned char* vc,const int* pt,
 const int* seqs,const long* loc,const float* ks,const float* vs,float* pa,float* pm,
 int HQ,int HKV,int MP,int S,float scale){
 constexpr int NS=3;
 extern __shared__ __align__(1024) unsigned char sm[];
 Shared& sh=*(Shared*)sm;
 auto& K=sh.K;auto& VR=sh.VR;auto& VT=sh.VT;auto& Q=sh.Q;auto& P=sh.P;auto& own=sh.own;auto& stats=sh.stats;
 const int tid=threadIdx.x,warp=tid/32,lane=tid%32;
 const int b=blockIdx.y,h0=blockIdx.z*2,split=blockIdx.x;
 const int seq=seqs[b],width=(2*HQ+2*HKV)*64;
 const int chunk=((seq+1+64*S-1)/(64*S))*64,lo=split*chunk,hi=min(seq+1,lo+chunk);
 const int tiles=max(0,(hi-lo+63)/64);
 float kscale=*ks,vscale=*vs;scale*=kscale;
 asm volatile("griddepcontrol.launch_dependents;":::"memory");
 #pragma unroll
 for(int h=0;h<2;h++){
  #pragma unroll
  for(int j=0;j<2;j++){
   int head=warp+j*4;const __nv_bfloat16* p=qkv+b*width+((h0+h)*8+head)*64;
   float x=__bfloat162float(p[lane]),y=__bfloat162float(p[lane+32]);
   float inv=rsqrtf(sum32(x*x+y*y)/64.f+1e-5f);
   Q[h][sw64(head*64+lane)]=to8(x*inv);Q[h][sw64(head*64+lane+32)]=to8(y*inv);
  }
 }
 if(seq>=lo&&seq<hi&&warp<2){
  int h=warp;const __nv_bfloat16* p=qkv+b*width+(HQ+h0+h)*64;
  float x=__bfloat162float(p[lane]),y=__bfloat162float(p[lane+32]);
  float inv=rsqrtf(sum32(x*x+y*y)/64.f+1e-5f)/kscale;
  unsigned char a=to8(x*inv),bb=to8(y*inv);
  unsigned char c=to8(__bfloat162float(p[HKV*64+lane])/vscale),d=to8(__bfloat162float(p[HKV*64+lane+32])/vscale);
  own[h][lane]=a;own[h][lane+32]=bb;own[h][lane+64]=c;own[h][lane+96]=d;
  long dst=loc[b]*HKV*64+(h0+h)*64;
  kc[dst+lane]=a;kc[dst+lane+32]=bb;vc[dst+lane]=c;vc[dst+lane+32]=d;
 }
 auto gather=[&](int tile){
  int slot=tile%NS;
  #pragma unroll
  for(int j=0;j<4;j++){
   int i=tid+j*128,row=i/8,h=(i%8)/4,piece=i%4*16,t=lo+tile*64+row;
   bool valid=t<hi&&t<seq;
   int pg=valid?pt[b*MP+t/PAGE]:0;
   long addr=((long)pg*PAGE+t%PAGE)*HKV*64+(h0+h)*64+piece;
   cp16(K[slot][h]+sw64(row*64+piece),kc+(valid?addr:0),valid);
   cp16(VR[slot][h]+row*64+piece,vc+(valid?addr:0),valid);
  }
  asm volatile("cp.async.commit_group;":::"memory");
 };
 gather(0);gather(1);
 float acc[2][4]={{0}},mx[2][2]={{-1e30f,-1e30f},{-1e30f,-1e30f}},sum[2][2]={{0}};
 for(int tile=0;tile<tiles;tile++){
  gather(tile+2);int slot=tile%NS;
  asm volatile("cp.async.wait_group 2;":::"memory");__syncthreads();
  if(seq>=lo+tile*64&&seq<min(hi,lo+(tile+1)*64)){
   if(warp<2){int row=seq-lo-tile*64,h=warp;
    K[slot][h][sw64(row*64+lane)]=own[h][lane];K[slot][h][sw64(row*64+lane+32)]=own[h][lane+32];
    VR[slot][h][row*64+lane]=own[h][lane+64];VR[slot][h][row*64+lane+32]=own[h][lane+96];
   }__syncthreads();
  }
  // Shared stores of Q and the new key must be visible to the async proxy.
  asm volatile("fence.proxy.async.shared::cta;":::"memory");__syncthreads();
  float score[2][4]={{0}};
  pin(score[0]);pin(score[1]);asm volatile("wgmma.fence.sync.aligned;":::"memory");
  #pragma unroll
  for(int h=0;h<2;h++){
   #pragma unroll
   for(int d=0;d<2;d++)qk(score[h],desc64(K[slot][h]+d*32),desc64(Q[h]+d*32));
  }
  asm volatile("wgmma.commit_group.sync.aligned;wgmma.wait_group.sync.aligned 0;":::"memory");pin(score[0]);pin(score[1]);
  #pragma unroll
  for(int h=0;h<2;h++){
   #pragma unroll
   for(int j=0;j<4;j++){
    unsigned i=8*tid+j*1024,target=sw128(i);
    unsigned long long bytes=*(unsigned long long*)(VR[slot][h]+i);unsigned cv[4];
    #pragma unroll
    for(int z=0;z<4;z++) {__half2_raw v=__nv_cvt_fp8x2_to_halfraw2((unsigned short)(bytes>>(16*z)),__NV_E4M3);cv[z]=*(unsigned*)&v;}
    asm volatile("st.shared.v4.u32 [%4],{%0,%1,%2,%3};"::"r"(cv[0]),"r"(cv[1]),"r"(cv[2]),"r"(cv[3]),"r"((unsigned)__cvta_generic_to_shared(VT[h]+target)):"memory");
   }
   #pragma unroll
   for(int c=0;c<2;c++){
    float m=-1e30f;
    #pragma unroll
    for(int r=0;r<2;r++){
     int row=warp*16+lane/4+r*8;
     score[h][r*2+c]=lo+tile*64+row<hi?score[h][r*2+c]*scale:-1e30f;
     m=fmaxf(m,score[h][r*2+c]);
    }
    #pragma unroll
    for(int d=4;d<=16;d*=2)m=fmaxf(m,__shfl_xor_sync(0xffffffff,m,d));
    if(lane<4)stats[h][warp][lane*2+c]=m;
   }
  }
  __syncthreads();
  #pragma unroll
  for(int h=0;h<2;h++){
   #pragma unroll
   for(int c=0;c<2;c++){
    int head=(lane%4)*2+c;float nm=mx[h][c];
    #pragma unroll
    for(int w=0;w<4;w++)nm=fmaxf(nm,stats[h][w][head]);
    float alpha=exp2f(mx[h][c]-nm);mx[h][c]=nm;sum[h][c]*=alpha;
    acc[h][c]*=alpha;acc[h][c+2]*=alpha;
    #pragma unroll
    for(int r=0;r<2;r++){
     int row=warp*16+lane/4+r*8;
     float p=lo+tile*64+row<hi?exp2f(score[h][r*2+c]-nm):0.f;
     score[h][r*2+c]=p;P[h][sw128(head*64+row)]=__float2half_rn(p);
    }
   }
  }
  // All threads finish reading maxima before reusing stats for sums.
  __syncthreads();
  #pragma unroll
  for(int h=0;h<2;h++){
   #pragma unroll
   for(int c=0;c<2;c++){
    float s=score[h][c]+score[h][c+2];
    #pragma unroll
    for(int d=4;d<=16;d*=2)s+=__shfl_xor_sync(0xffffffff,s,d);
    if(lane<4)stats[h][warp][lane*2+c]=s;
   }
  }
  asm volatile("fence.proxy.async.shared::cta;":::"memory");__syncthreads();
  pin(acc[0]);pin(acc[1]);asm volatile("wgmma.fence.sync.aligned;":::"memory");
  #pragma unroll
  for(int h=0;h<2;h++){
   #pragma unroll
   for(int d=0;d<4;d++)pv(acc[h],((((unsigned long)__cvta_generic_to_shared(VT[h]+d*1024)&0x3ffff)>>4)|(64ull<<16)|(64ull<<32)|(1ull<<62)),desc128(P[h]+sw128(d*16)));
  }
  asm volatile("wgmma.commit_group.sync.aligned;wgmma.wait_group.sync.aligned 0;":::"memory");pin(acc[0]);pin(acc[1]);
  #pragma unroll
  for(int h=0;h<2;h++){
   #pragma unroll
   for(int c=0;c<2;c++){
    #pragma unroll
    for(int w=0;w<4;w++)sum[h][c]+=stats[h][w][lane%4*2+c];
   }
  }
 }
 asm volatile("cp.async.wait_group 0;":::"memory");__syncthreads();
 #pragma unroll
 for(int h=0;h<2;h++){
  long base=((long)(b*HKV+h0+h)*S+split)*8;
  #pragma unroll
  for(int c=0;c<2;c++){
   int head=lane%4*2+c;
   #pragma unroll
   for(int r=0;r<2;r++)pa[(base+head)*64+warp*16+lane/4+r*8]=acc[h][r*2+c];
   if(warp==0&&lane<4){pm[(base+head)*2]=mx[h][c];pm[(base+head)*2+1]=sum[h][c];}
  }
 }
}
void attention(at::Tensor qkv,at::Tensor kc,at::Tensor vc,at::Tensor pt,at::Tensor seq,
 at::Tensor loc,at::Tensor ks,at::Tensor vs,at::Tensor pa,at::Tensor pm,int page,double scale){
 int hkv=kc.size(2),hq=(qkv.size(2)/64-2*hkv)/2,S=pa.size(2);
 dim3 grid(S,qkv.size(0),hkv/2);auto stream=c10::cuda::getCurrentCUDAStream();
 #define RUN(P) cudaFuncSetAttribute(decode<P>,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared));decode<P><<<grid,128,sizeof(Shared),stream>>>((const __nv_bfloat16*)qkv.data_ptr(),(unsigned char*)kc.data_ptr(),(unsigned char*)vc.data_ptr(),pt.data_ptr<int>(),seq.data_ptr<int>(),loc.data_ptr<long>(),ks.data_ptr<float>(),vs.data_ptr<float>(),pa.data_ptr<float>(),pm.data_ptr<float>(),hq,hkv,pt.size(1),S,scale)
 if(page==1){RUN(1);}else{RUN(64);}
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}

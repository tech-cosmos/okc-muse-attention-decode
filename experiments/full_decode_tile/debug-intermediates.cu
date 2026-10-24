#include <ATen/core/Tensor.h>
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
 __shared__ __align__(512) unsigned char K[8192];
 __shared__ __align__(16) unsigned char Vraw[8192];
 __shared__ __align__(1024) half V[8192];
 const int tid=threadIdx.x,warp=tid/32,lane=tid%32;
 const int b=blockIdx.y,h=blockIdx.z,split=blockIdx.x;
 const int seq=seqs[b],base=0,lo=split*128;
 const int width=(2*HQ+2*HKV)*64;
 asm volatile("griddepcontrol.launch_dependents;":::"memory");
 // Each thread starts two 16B copies per cache, over all 64 positions.
 auto gather=[&]() {
  #pragma unroll
  for(int j=0;j<8;j++) {
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

 unsigned qr[2][4]={{0}};
 if(warp==0) {
  float x[16],sq=0;
  #pragma unroll
  for(int i=0;i<16;i++) {
   int col=(i/8)*32+(lane%4)*4+((i%8)/4)*16+i%4;
   x[i]=__bfloat162float(qkv[b*width+(h*8+lane/4)*64+col]);sq+=x[i]*x[i];
  }
  sq+=__shfl_xor_sync(0xffffffff,sq,1);sq+=__shfl_xor_sync(0xffffffff,sq,2);
  float inv=rsqrtf(sq/64.f+1e-5f);
  #pragma unroll
  for(int d=0;d<2;d++) {
   #pragma unroll
   for(int i=0;i<2;i++) {
    unsigned packed=0;
    #pragma unroll
    for(int k=0;k<4;k+=2) {
     int col=d*32+(lane%4)*4+i*16+k,ix=d*8+i*4+k;
     float a=x[ix]*inv,bb=x[ix+1]*inv;
     packed|=(unsigned)to8(a)<<(8*k);packed|=(unsigned)to8(bb)<<(8*(k+1));
    }
    qr[d][i*2]=packed;
   }
  }
 }
 // Keep new K/V in registers until the zero-filled cache gather completes.
 float nx=0,ny=0,nv0=0,nv1=0;
 bool own=seq>=lo && seq<lo+128;
 if(own && warp==1) {
  const __nv_bfloat16* p=qkv+b*width+(HQ+h)*64;
  nx=__bfloat162float(p[lane]);ny=__bfloat162float(p[lane+32]);
  float inv=rsqrtf(sum32(nx*nx+ny*ny)/64.f+1e-5f)/kscale;
  nx*=inv;ny*=inv;
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
 for(int j=0;j<8;j++) {
  unsigned i=8*tid+j*1024,target=i ^ (((i>>6)&7)<<3);
  unsigned long long bytes=*(unsigned long long*)(Vraw+i);unsigned cv[4];
  #pragma unroll
  for(int z=0;z<4;z++) {__half2_raw v=__nv_cvt_fp8x2_to_halfraw2((unsigned short)(bytes>>(16*z)),__NV_E4M3);cv[z]=*(unsigned*)&v;}
  asm volatile("st.shared.v4.u32 [%4],{%0,%1,%2,%3};"::"r"(cv[0]),"r"(cv[1]),"r"(cv[2]),"r"(cv[3]),"r"((unsigned)__cvta_generic_to_shared(V+target)):"memory");
 }
 asm volatile("fence.proxy.async.shared::cta;":::"memory");
 __syncthreads();
 float score[64]={0};
 asm volatile("" : "+f"(score[0]),"+f"(score[1]),"+f"(score[2]),"+f"(score[3]),"+f"(score[4]),"+f"(score[5]),"+f"(score[6]),"+f"(score[7]),"+f"(score[8]),"+f"(score[9]),"+f"(score[10]),"+f"(score[11]),"+f"(score[12]),"+f"(score[13]),"+f"(score[14]),"+f"(score[15]),"+f"(score[16]),"+f"(score[17]),"+f"(score[18]),"+f"(score[19]),"+f"(score[20]),"+f"(score[21]),"+f"(score[22]),"+f"(score[23]),"+f"(score[24]),"+f"(score[25]),"+f"(score[26]),"+f"(score[27]),"+f"(score[28]),"+f"(score[29]),"+f"(score[30]),"+f"(score[31]),"+f"(score[32]),"+f"(score[33]),"+f"(score[34]),"+f"(score[35]),"+f"(score[36]),"+f"(score[37]),"+f"(score[38]),"+f"(score[39]),"+f"(score[40]),"+f"(score[41]),"+f"(score[42]),"+f"(score[43]),"+f"(score[44]),"+f"(score[45]),"+f"(score[46]),"+f"(score[47]),"+f"(score[48]),"+f"(score[49]),"+f"(score[50]),"+f"(score[51]),"+f"(score[52]),"+f"(score[53]),"+f"(score[54]),"+f"(score[55]),"+f"(score[56]),"+f"(score[57]),"+f"(score[58]),"+f"(score[59]),"+f"(score[60]),"+f"(score[61]),"+f"(score[62]),"+f"(score[63]) :: "memory");
 asm volatile("wgmma.fence.sync.aligned;":::"memory");
 #pragma unroll
 for(int d=0;d<2;d++) {
  unsigned long desc=(((unsigned long)__cvta_generic_to_shared(K+d*32)&0x3ffff)>>4)|(1ull<<16)|(32ull<<32)|(2ull<<62);
  asm volatile("wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, {%64,%65,%66,%67}, %68, 1, 1, 1;" : "+f"(score[0]),"+f"(score[1]),"+f"(score[2]),"+f"(score[3]),"+f"(score[4]),"+f"(score[5]),"+f"(score[6]),"+f"(score[7]),"+f"(score[8]),"+f"(score[9]),"+f"(score[10]),"+f"(score[11]),"+f"(score[12]),"+f"(score[13]),"+f"(score[14]),"+f"(score[15]),"+f"(score[16]),"+f"(score[17]),"+f"(score[18]),"+f"(score[19]),"+f"(score[20]),"+f"(score[21]),"+f"(score[22]),"+f"(score[23]),"+f"(score[24]),"+f"(score[25]),"+f"(score[26]),"+f"(score[27]),"+f"(score[28]),"+f"(score[29]),"+f"(score[30]),"+f"(score[31]),"+f"(score[32]),"+f"(score[33]),"+f"(score[34]),"+f"(score[35]),"+f"(score[36]),"+f"(score[37]),"+f"(score[38]),"+f"(score[39]),"+f"(score[40]),"+f"(score[41]),"+f"(score[42]),"+f"(score[43]),"+f"(score[44]),"+f"(score[45]),"+f"(score[46]),"+f"(score[47]),"+f"(score[48]),"+f"(score[49]),"+f"(score[50]),"+f"(score[51]),"+f"(score[52]),"+f"(score[53]),"+f"(score[54]),"+f"(score[55]),"+f"(score[56]),"+f"(score[57]),"+f"(score[58]),"+f"(score[59]),"+f"(score[60]),"+f"(score[61]),"+f"(score[62]),"+f"(score[63]) : "r"(qr[d][0]),"r"(qr[d][1]),"r"(qr[d][2]),"r"(qr[d][3]),"l"(desc));
 }
 asm volatile("wgmma.commit_group.sync.aligned;\nwgmma.wait_group.sync.aligned 0;":::"memory");
 asm volatile("" : "+f"(score[0]),"+f"(score[1]),"+f"(score[2]),"+f"(score[3]),"+f"(score[4]),"+f"(score[5]),"+f"(score[6]),"+f"(score[7]),"+f"(score[8]),"+f"(score[9]),"+f"(score[10]),"+f"(score[11]),"+f"(score[12]),"+f"(score[13]),"+f"(score[14]),"+f"(score[15]),"+f"(score[16]),"+f"(score[17]),"+f"(score[18]),"+f"(score[19]),"+f"(score[20]),"+f"(score[21]),"+f"(score[22]),"+f"(score[23]),"+f"(score[24]),"+f"(score[25]),"+f"(score[26]),"+f"(score[27]),"+f"(score[28]),"+f"(score[29]),"+f"(score[30]),"+f"(score[31]),"+f"(score[32]),"+f"(score[33]),"+f"(score[34]),"+f"(score[35]),"+f"(score[36]),"+f"(score[37]),"+f"(score[38]),"+f"(score[39]),"+f"(score[40]),"+f"(score[41]),"+f"(score[42]),"+f"(score[43]),"+f"(score[44]),"+f"(score[45]),"+f"(score[46]),"+f"(score[47]),"+f"(score[48]),"+f"(score[49]),"+f"(score[50]),"+f"(score[51]),"+f"(score[52]),"+f"(score[53]),"+f"(score[54]),"+f"(score[55]),"+f"(score[56]),"+f"(score[57]),"+f"(score[58]),"+f"(score[59]),"+f"(score[60]),"+f"(score[61]),"+f"(score[62]),"+f"(score[63]) :: "memory");

 if(warp==0)for(int i=0;i<64;i++)if(!(i&2)) {
  int row=lane/4,col=(i/8)*16+(lane%4)*2+(i&1)+(i&4)*2;
  pa[((b*HKV+h)*S+split)*8*128+row*128+col]=score[i];
 }
 for(int i=tid;i<8192;i+=128) {
  unsigned short vv;
  asm volatile("ld.shared.u16 %0,[%1];" : "=h"(vv) : "r"((unsigned)__cvta_generic_to_shared(V+(i ^ (((i>>6)&7)<<3)))) : "memory");
  pm[((b*HKV+h)*S+split)*8192+i]=__half2float(*(__half*)&vv);
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

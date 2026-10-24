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

 unsigned qr[2][4]={{0}};
 for(int r=0;r<2;r++) {
  int token=warp*2+r;
  float x[16],sq=0;
  #pragma unroll
  for(int i=0;i<4;i++) {
   int col=(i/2)*32+(lane%4)*4+(i%2)*16;
   const __nv_bfloat16* p=qkv+(b*8+token)*width+(h*8+lane/4)*64+col;
   unsigned a,bits;
   asm volatile("ld.global.v2.u32 {%0,%1},[%2];":"=r"(a),"=r"(bits):"l"(p):"memory");
   x[4*i]=__bfloat162float(__ushort_as_bfloat16(a&65535));
   x[4*i+1]=__bfloat162float(__ushort_as_bfloat16(a>>16));
   x[4*i+2]=__bfloat162float(__ushort_as_bfloat16(bits&65535));
   x[4*i+3]=__bfloat162float(__ushort_as_bfloat16(bits>>16));
  }
  #pragma unroll
  for(int i=0;i<16;i++)sq+=x[i]*x[i];
  sq+=__shfl_xor_sync(0xffffffff,sq,1);sq+=__shfl_xor_sync(0xffffffff,sq,2);
  float inv=rsqrtf(sq/64.f+1e-5f);
  #pragma unroll
  for(int d=0;d<2;d++) {
   #pragma unroll
   for(int i=0;i<2;i++) {
    unsigned packed=0;
    int cscol=d*16+(lane%4)*2+i*8;
    const float* csp=cs+(long)(seq+token)*64+cscol;
    float cc0,cc1,ss0,ss1;
    asm volatile("ld.global.v2.f32 {%0,%1},[%2];":"=f"(cc0),"=f"(cc1):"l"(csp):"memory");
    asm volatile("ld.global.v2.f32 {%0,%1},[%2];":"=f"(ss0),"=f"(ss1):"l"(csp+32):"memory");
    #pragma unroll
    for(int k=0;k<4;k+=2) {
     int col=d*32+(lane%4)*4+i*16+k,ix=d*8+i*4+k;
     float cc=k==0?cc0:cc1,ss=k==0?ss0:ss1;
     float a=(x[ix]*cc-x[ix+1]*ss)*inv,bb=(x[ix+1]*cc+x[ix]*ss)*inv;
     packed|=(unsigned)to8(a)<<(8*k);packed|=(unsigned)to8(bb)<<(8*(k+1));
    }
    qr[d][i*2+r]=packed;
   }
  }
 }
 // Each warp prepares up to two new tokens; only their natural tile writes them.
 float nx[2]={0,0},ny[2]={0,0},nv0[2]={0,0},nv1[2]={0,0};
 bool own[2];
 #pragma unroll
 for(int j=0;j<2;j++) {
  int token=warp+4*j,pos=seq+token;
  own[j]=pos>=lo && pos<lo+64;
  if(own[j]) {
   const __nv_bfloat16* p=qkv+(b*8+token)*width+(HQ+h)*64;
   nx[j]=__bfloat162float(p[lane]);ny[j]=__bfloat162float(p[lane+32]);
   float inv=rsqrtf(sum32(nx[j]*nx[j]+ny[j]*ny[j])/64.f+1e-5f)/kscale;
   float xp=__shfl_xor_sync(0xffffffff,nx[j],1),yp=__shfl_xor_sync(0xffffffff,ny[j],1),sign=lane%2?1.f:-1.f;
   float c0=cs[(long)pos*64+lane/2],s0=cs[(long)pos*64+lane/2+32];
   float c1=cs[(long)pos*64+(lane+32)/2],s1=cs[(long)pos*64+(lane+32)/2+32];
   nx[j]=(nx[j]*c0+sign*xp*s0)*inv;ny[j]=(ny[j]*c1+sign*yp*s1)*inv;
   nv0[j]=__bfloat162float(p[HKV*64+lane])/vscale;nv1[j]=__bfloat162float(p[HKV*64+lane+32])/vscale;
  }
 }
 if(!EARLY)gather();
 asm volatile("cp.async.wait_group 0;":::"memory");
 __syncthreads();
 #pragma unroll
 for(int j=0;j<2;j++)if(own[j]) {
  int token=warp+4*j,row=seq+token-lo;
  unsigned char k0=to8(nx[j]),k1=to8(ny[j]),v0=to8(nv0[j]),v1=to8(nv1[j]);
  K[sw64(row*64+lane)]=k0;K[sw64(row*64+lane+32)]=k1;
  Vraw[row*64+lane]=v0;Vraw[row*64+lane+32]=v1;
  long slot=loc[b*8+token]*HKV*64+h*64;
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
  int pos=seq+warp*2+(i&2)/2;
  score[i]=(lo+col<=pos && lo+col>=pos-2047)?score[i]:-1e30f;
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
  int pos=seq+warp*2+r;
  float p=(lo+col<=pos && lo+col>=pos-2047)?exp2f(fmaf(score[i],scale*kscale,-mx[r])):0.f;
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
 long outbase=((long)(b*HKV+h)*S+split)*64;
 #pragma unroll
 for(int i=0;i<32;i++) {
  int row=warp*16+lane/4+((i&2)/2)*8,col=(i/8)*16+(lane%4)*2+(i&1)+(i&4)*2;
  pa[(outbase+row)*64+col]=acc[i];
 }
 if(lane%4==0) {
  #pragma unroll
  for(int r=0;r<2;r++) {
   int row=warp*16+lane/4+r*8;
   pm[(outbase+row)*2]=mx[r];pm[(outbase+row)*2+1]=sum[r];
  }
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

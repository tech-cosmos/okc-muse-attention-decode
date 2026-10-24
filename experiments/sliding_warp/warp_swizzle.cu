#define THREADS 32
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
__global__ __launch_bounds__(THREADS) void sliding(const __nv_bfloat16* qkv,const float* cs,
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
  for(int j=0;j<256/THREADS;j++) {
   int i=tid+j*THREADS,row=i/4,piece=(i%4)*16,t=lo+row;
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
     float cc=cs[(long)seq*64+col/2],ss=cs[(long)seq*64+col/2+32];
     float a=(x[ix]*cc-x[ix+1]*ss)*inv,bb=(x[ix+1]*cc+x[ix]*ss)*inv;
     packed|=(unsigned)to8(a)<<(8*k);packed|=(unsigned)to8(bb)<<(8*(k+1));
    }
    qr[d][i*2]=packed;
   }
  }
 }
 // Keep new K/V in registers until the zero-filled cache gather completes.
 float nx=0,ny=0,nv0=0,nv1=0;
 bool own=seq>=lo && seq<lo+64;
 if(own && warp==(THREADS==32?0:1)) {
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
 if(own && warp==(THREADS==32?0:1)) {
  int row=seq-lo;unsigned char k0=to8(nx),k1=to8(ny),v0=to8(nv0),v1=to8(nv1);
  K[sw64(row*64+lane)]=k0;K[sw64(row*64+lane+32)]=k1;
  Vraw[row*64+lane]=v0;Vraw[row*64+lane+32]=v1;
  long slot=loc[b]*HKV*64+h*64;
  kc[slot+lane]=k0;kc[slot+lane+32]=k1;vc[slot+lane]=v0;vc[slot+lane+32]=v1;
 }
 __syncthreads();
 #pragma unroll
 for(int j=0;j<512/THREADS;j++) {
  unsigned i=8*tid+j*(8*THREADS),target=i ^ (((i>>6)&7)<<3);
  unsigned long long bytes=*(unsigned long long*)(Vraw+i);unsigned cv[4];
  #pragma unroll
  for(int z=0;z<4;z++) {__half2_raw v=__nv_cvt_fp8x2_to_halfraw2((unsigned short)(bytes>>(16*z)),__NV_E4M3);cv[z]=*(unsigned*)&v;}
  asm volatile("st.shared.v4.u32 [%4],{%0,%1,%2,%3};"::"r"(cv[0]),"r"(cv[1]),"r"(cv[2]),"r"(cv[3]),"r"((unsigned)__cvta_generic_to_shared(V+target)):"memory");
 }
 asm volatile("fence.proxy.async.shared::cta;":::"memory");
 __syncthreads();

 if(warp==0) {
 float score[32]={0};
 #pragma unroll
 for(int n=0;n<8;n++) {
  #pragma unroll
  for(int d=0;d<2;d++) {
   unsigned kr0=*(unsigned*)(K+sw64((n*8+lane/4)*64+d*32+(lane%4)*4));
   unsigned kr1=*(unsigned*)(K+sw64((n*8+lane/4)*64+d*32+(lane%4)*4+16));
   asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    : "+f"(score[n*4]),"+f"(score[n*4+1]),"+f"(score[n*4+2]),"+f"(score[n*4+3])
    : "r"(qr[d][0]),"r"(qr[d][1]),"r"(qr[d][2]),"r"(qr[d][3]),"r"(kr0),"r"(kr1));
  }
 }
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
 #pragma unroll
 for(int n=0;n<8;n++) {
  #pragma unroll
  for(int d=0;d<4;d++) {
   unsigned vr0,vr1;
   unsigned vi=(d*16+lane%16)*64+n*8;
   unsigned addr=(unsigned)__cvta_generic_to_shared(V+(vi ^ (((vi>>6)&7)<<3)));
   asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];":"=r"(vr0),"=r"(vr1):"r"(addr));
   asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    : "+f"(acc[n*4]),"+f"(acc[n*4+1]),"+f"(acc[n*4+2]),"+f"(acc[n*4+3])
    : "r"(pr[d][0]),"r"(pr[d][1]),"r"(pr[d][2]),"r"(pr[d][3]),"r"(vr0),"r"(vr1));
  }
 }
 if(warp==0) {
  long outbase=((long)(b*HKV+h)*S+split)*8;
  #pragma unroll
  for(int i=0;i<32;i++)if(!(i&2)) {
   int row=lane/4,col=(i/8)*16+(lane%4)*2+(i&1)+(i&4)*2;
   pa[(outbase+row)*64+col]=acc[i];
  }
  if(lane%4==0){pm[(outbase+lane/4)*2]=mx[0];pm[(outbase+lane/4)*2+1]=sum[0];}
 }
 } // warp zero arithmetic
}
void attention(at::Tensor qkv,at::Tensor cs,at::Tensor kc,at::Tensor vc,at::Tensor pt,at::Tensor seq,
 at::Tensor loc,at::Tensor ks,at::Tensor vs,at::Tensor pa,at::Tensor pm,int page,double scale,bool early) {
 int hkv=kc.size(2),hq=(qkv.size(2)/64-2*hkv)/2,S=pa.size(2);
 dim3 grid(S,qkv.size(0),hkv);auto stream=c10::cuda::getCurrentCUDAStream();
 #define RUN(P,E) sliding<P,E><<<grid,THREADS,0,stream>>>((const __nv_bfloat16*)qkv.data_ptr(),cs.data_ptr<float>(),(unsigned char*)kc.data_ptr(),(unsigned char*)vc.data_ptr(),pt.data_ptr<int>(),seq.data_ptr<int>(),loc.data_ptr<long>(),ks.data_ptr<float>(),vs.data_ptr<float>(),pa.data_ptr<float>(),pm.data_ptr<float>(),hq,hkv,pt.size(1),S,scale)
 if(page==1){if(early){RUN(1,true);}else{RUN(1,false);}}else{if(early){RUN(64,true);}else{RUN(64,false);}}
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}

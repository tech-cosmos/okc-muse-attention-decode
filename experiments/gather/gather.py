"""Native full-row gather ceiling: real page-table reads, K/V reads, byte checksums.
This is a diagnostic only, never an attention submission. Graph timing comes from bench.py.
"""
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline
CPP=r'''
#include <torch/extension.h>
void gather(torch::Tensor k,torch::Tensor v,torch::Tensor pt,torch::Tensor seq,torch::Tensor out,int splits,int heads,int stages,int producer);
'''
CUDA=r'''
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
__device__ void cp(void* d,const void* s){asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"((unsigned)__cvta_generic_to_shared(d)),"l"(s));}
__device__ unsigned acquire(unsigned* p){unsigned v;asm volatile("ld.acquire.cta.shared.u32 %0,[%1];":"=r"(v):"r"((unsigned)__cvta_generic_to_shared(p)):"memory");return v;}
__device__ void release(unsigned* p,unsigned v){asm volatile("st.release.cta.shared.u32 [%0],%1;"::"r"((unsigned)__cvta_generic_to_shared(p)),"r"(v):"memory");}
template<int H,int ST,int PW>
__global__ void gather_kernel(const unsigned char* k,const unsigned char* v,const int* pt,const int* seq,unsigned* out,int MP,int HKV,int S,int PAGE){
 constexpr int N=32,D=64*H,THREADS=128; constexpr bool PROD=PW>0;
 __shared__ __align__(16) unsigned char raw[ST][2][N*D];
 __shared__ unsigned ready[ST];
 unsigned tid=threadIdx.x,lane=tid%32,pload=tid-128;
 int b=blockIdx.y,h=blockIdx.z*H,s=blockIdx.x;
 int chunk=((seq[b]+N*S-1)/(N*S))*N,lo=s*chunk,hi=min(seq[b],lo+chunk),tiles=max(0,(hi-lo+N-1)/N);
 if(tid<ST)ready[tid]=0;
 __syncthreads();
 unsigned sum=0;
 if(PROD && tid>=THREADS){
  for(int step=0;step<tiles+ST-1;step++){
   if(step>=ST-1){
    if(step<tiles){asm volatile("cp.async.wait_group %0;"::"n"(ST-2):"memory");}
    else {asm volatile("cp.async.wait_group 0;":::"memory");}
    asm volatile("bar.sync 2,%0;"::"r"(PW*32):"memory");if(pload==0)release(ready+(step-ST+1)%ST,1);
   }
   if(step<tiles){
    int stage=step%ST;while(acquire(ready+stage)!=0){}
    for(int i=pload;i<N*D/16;i+=PW*32){
     int t=min(lo+step*N+i/(D/16),hi-1);
     long row=((long)pt[b*MP+t/PAGE]*PAGE+t%PAGE)*HKV*64+h*64;
     cp(raw[stage][0]+i*16,k+row+i%(D/16)*16);cp(raw[stage][1]+i*16,v+row+i%(D/16)*16);
    }
    asm volatile("cp.async.commit_group;":::"memory");
   }
  }
  return;
 }
 if(PROD){
  for(int tile=0;tile<tiles;tile++){
   int st=tile%ST;while(acquire(ready+st)!=1){}
   for(int i=tid;i<N*D/4;i+=THREADS){
    if(lo+tile*N+i/(D/4)<hi)sum+=((unsigned*)raw[st][0])[i]+((unsigned*)raw[st][1])[i];
   }
   asm volatile("bar.sync 1,128;":::"memory");if(tid==0)release(ready+st,0);
  }
 }else{
  for(int step=0;step<tiles+ST-1;step++){
   if(step>=ST-1){
    if(step<tiles){asm volatile("cp.async.wait_group %0;"::"n"(ST-2):"memory");}
    else {asm volatile("cp.async.wait_group 0;":::"memory");}
    __syncthreads();int tile=step-ST+1,st=tile%ST;
    for(int i=tid;i<N*D/4;i+=THREADS){
     if(lo+tile*N+i/(D/4)<hi)sum+=((unsigned*)raw[st][0])[i]+((unsigned*)raw[st][1])[i];
    }
    __syncthreads();
   }
   if(step<tiles){
    int stage=step%ST;
    for(int i=tid;i<N*D/16;i+=THREADS){
     int t=min(lo+step*N+i/(D/16),hi-1);
     long row=((long)pt[b*MP+t/PAGE]*PAGE+t%PAGE)*HKV*64+h*64;
     cp(raw[stage][0]+i*16,k+row+i%(D/16)*16);cp(raw[stage][1]+i*16,v+row+i%(D/16)*16);
    }
    asm volatile("cp.async.commit_group;":::"memory");
   }
  }
 }
 out[(((b*(HKV/H)+blockIdx.z)*S+s)*THREADS)+tid]=sum;
}
void gather(at::Tensor k,at::Tensor v,at::Tensor pt,at::Tensor seq,at::Tensor out,int splits,int heads,int stages,int producer){
 dim3 grid(splits,pt.size(0),k.size(2)/heads);auto stream=c10::cuda::getCurrentCUDAStream();
 #define RUN(H,ST,P) gather_kernel<H,ST,P><<<grid,128+32*P,0,stream>>>((const unsigned char*)k.data_ptr(),(const unsigned char*)v.data_ptr(),pt.data_ptr<int>(),seq.data_ptr<int>(),(unsigned*)out.data_ptr(),pt.size(1),k.size(2),splits,k.size(1))
 #define PP(H,ST) if(producer==1){RUN(H,ST,1);}else if(producer==2){RUN(H,ST,2);}else if(producer==4){RUN(H,ST,4);}else{RUN(H,ST,0);}
 #define SS(H) if(stages==2){PP(H,2);}else if(stages==3){PP(H,3);}else if(stages==4){PP(H,4);}else{PP(H,6);}
 if(heads==1){SS(1);}else{SS(2);}
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''
build=Path(__file__).parent/'.build-gather';build.mkdir(exist_ok=True)
module=load_inline(name='codex_gather3b',build_directory=str(build),cpp_sources=CPP,cuda_sources=CUDA,functions=['gather'],extra_cuda_cflags=['-gencode=arch=compute_90a,code=sm_90a','-O3','--ptxas-options=-v'],verbose=True)

def run(data,splits=16,heads=2,stages=3,producer=0):
    c=data['config'];out=torch.empty((c['batch'],c['num_kv_heads']//heads,splits,128),dtype=torch.int32,device='cuda')
    module.gather(data['k_cache'],data['v_cache'],data['page_table'],data['seq_lens'],out,splits,heads,stages,producer)
    return out

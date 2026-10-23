"""Native bit-exact paged FP8 gather, for the session-6 materialization experiment."""
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline
build = Path(__file__).parent / '.build'
build.mkdir(exist_ok=True)
mod = load_inline(
 name='muse_pack_s6', build_directory=str(build),
 cpp_sources=r'''
#include <torch/extension.h>
void pack(torch::Tensor k,torch::Tensor v,torch::Tensor pt,torch::Tensor seq,torch::Tensor ko,torch::Tensor vo,int page,int blocks,int unroll,int vec);
''',
 cuda_sources=r'''
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
template<int V> struct Vec;
template<> struct Vec<4>{using T=uint32_t;};
template<> struct Vec<8>{using T=uint2;};
template<> struct Vec<16>{using T=uint4;};
template<int V,int U>
__global__ void pack_kernel(const uint8_t* k,const uint8_t* v,const int* pt,const int* seq,
 uint8_t* ko,uint8_t* vo,int P,int MP,int C,int ROW) {
 using T=typename Vec<V>::T;
 unsigned b=blockIdx.y, tid=blockIdx.x*blockDim.x+threadIdx.x;
 unsigned cols=ROW/V,step=gridDim.x*blockDim.x, lim=(unsigned)seq[b]*cols;
 for(unsigned base=tid;base<lim;base+=step*U){
   T kval[U],vval[U];unsigned dst[U];bool valid[U];
   #pragma unroll
   for(int u=0;u<U;u++) {
     unsigned idx=base+u*step,row=idx/cols,col=idx%cols;
     valid[u]=idx<lim;dst[u]=((b*C+row)*ROW)/V+col;
     if(valid[u]) {
       unsigned physical=pt[b*MP+row/P]*P+row%P;
       kval[u]=((const T*)k)[physical*cols+col];
       vval[u]=((const T*)v)[physical*cols+col];
     }
   }
   #pragma unroll
   for(int u=0;u<U;u++)if(valid[u]) {
     ((T*)ko)[dst[u]]=kval[u];((T*)vo)[dst[u]]=vval[u];
   }
 }
}
void pack(at::Tensor k,at::Tensor v,at::Tensor pt,at::Tensor seq,at::Tensor ko,at::Tensor vo,int page,int blocks,int unroll,int vec) {
 auto stream=c10::cuda::getCurrentCUDAStream();
 int B=pt.size(0),MP=pt.size(1),C=ko.size(1),ROW=k.size(-2)*k.size(-1);
 dim3 grid(blocks,B);
 #define RUN(V,U) pack_kernel<V,U><<<grid,128,0,stream>>>((const uint8_t*)k.data_ptr(),(const uint8_t*)v.data_ptr(),pt.data_ptr<int>(),seq.data_ptr<int>(),(uint8_t*)ko.data_ptr(),(uint8_t*)vo.data_ptr(),page,MP,C,ROW)
 #define SEL(V) if(unroll==1){RUN(V,1);}else if(unroll==2){RUN(V,2);}else if(unroll==4){RUN(V,4);}else{RUN(V,8);}
 if(vec==4){SEL(4);}else if(vec==8){SEL(8);}else{SEL(16);}
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
''', functions=['pack'],extra_cuda_cflags=['-O3','-gencode=arch=compute_90a,code=sm_90a'],extra_cflags=['-O3'],verbose=False)

def allocate(data):
 cfg=data['config']; B=cfg['batch']; C=data['page_table'].shape[1]*cfg['page_size']
 shape=(B,C,cfg['num_kv_heads'],64)
 return (torch.empty(shape,device=data['qkv'].device,dtype=torch.float8_e4m3fn),
         torch.empty(shape,device=data['qkv'].device,dtype=torch.float8_e4m3fn))

def run(data,out,blocks=64,unroll=4,vec=16):
 mod.pack(data['k_cache'],data['v_cache'],data['page_table'],data['seq_lens'],*out,
          data['config']['page_size'],blocks,unroll,vec)

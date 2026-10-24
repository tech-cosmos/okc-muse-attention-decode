"""Independent CPU/GPU finite FP32 ordering audit, including subnormals and ties."""
import fcntl,json,sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.cpp_extension import load_inline
from generate import NEW
ROOT=Path(__file__).resolve().parent

def data():
 rng=np.random.default_rng(40091)
 bits=rng.integers(0,2**32,size=(32768,32),dtype=np.uint32)
 # Replace NaN/Inf exponents. Preserve every finite bit pattern sampled.
 bits[(bits&0x7f800000)==0x7f800000]=np.uint32(0xff7fffff)
 edge=np.array([0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,0x7f7fffff,0xff7fffff,0x3f800000,0xbf800000],dtype=np.uint32)
 rows=np.repeat(edge[:,None],32,axis=1)
 mixed=np.tile(edge,3)[:32][None,:].repeat(32,axis=0)
 for i in range(32):mixed[i]=np.roll(mixed[i],i)
 bits=np.concatenate([bits,rows,mixed])
 values=bits.view(np.float32)
 signed=bits.view(np.int32);ordered=signed^((signed>>31)&0x7fffffff)
 winner=ordered.max(1);inverse=winner^((winner>>31)&0x7fffffff)
 actual=inverse.view(np.float32);expected=values.max(1)
 assert np.array_equal(actual,expected)
 assert np.array_equal((ordered^((ordered>>31)&0x7fffffff)).view(np.uint32),bits)
 return values.copy(),expected.copy()

def run():
 values,expected=data()
 cpp='''#include <torch/extension.h>\nvoid probe(torch::Tensor x,torch::Tensor y);'''
 cuda='''#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
'''+NEW+'''
__global__ void probe_kernel(const float* x,float* y,int n) {
 int row=(blockIdx.x*blockDim.x+threadIdx.x)/32;
 if(row<n){float result=max32(x[row*32+threadIdx.x%32]);if(threadIdx.x%32==0)y[row]=result;}
}
void probe(at::Tensor x,at::Tensor y){
 probe_kernel<<<(x.size(0)+7)/8,256,0,c10::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(),y.data_ptr<float>(),x.size(0));
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''
 mod=load_inline(name='muse_redux_bitprobe_s40',cpp_sources=cpp,cuda_sources=cuda,functions=['probe'],extra_cuda_cflags=['-gencode=arch=compute_90a,code=sm_90a','-O3','--use_fast_math'])
 x=torch.tensor(values,device='cuda');y=torch.empty(len(values),device='cuda')
 mod.probe(x,y);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):mod.probe(x,y)
 for rep in range(3):
  # Rotate lanes without changing row maxima, at the same captured addresses.
  x.copy_(torch.tensor(np.roll(values,rep,axis=1).copy(),device='cuda'));g.replay()
  actual=y.cpu().numpy();assert np.array_equal(actual,expected),(rep,np.where(actual!=expected))
 record=dict(rows=len(values),finite_values=values.size,replays=3,exact_numeric_max=True,exact_cpu_bit_roundtrip=True,nan_inputs=False,signed_zero_bit_identity_claim=False)
 (ROOT/'bitprobe.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
if __name__=='__main__':
 with open('/tmp/okc-gpu.lock','w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);run()

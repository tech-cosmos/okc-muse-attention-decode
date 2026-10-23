"""Explore cooperative warp-zero weights for the final shared reduction."""
from pathlib import Path
import ast
from torch.utils.cpp_extension import load_inline
mod=ast.parse(Path('submission.py').read_text())
call=next(n.value for n in mod.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_NATIVE_MERGE' for t in n.targets))
s=next(ast.literal_eval(x.value) for x in call.keywords if x.arg=='cuda_sources')
a=s.index('   float gm=-1e30f;');b=s.index('\n }\n x=x/den*vscale;',a)
s=s[:a]+r'''
   float gm=partial[lane%W][64];
   #pragma unroll
   for(int d=W/2;d;d/=2)gm=fmaxf(gm,__shfl_xor_sync(0xffffffff,gm,d));
   float weight=exp2f(partial[lane%W][64]-gm);
   den=weight*partial[lane%W][65];
   #pragma unroll
   for(int d=W/2;d;d/=2)den+=__shfl_xor_sync(0xffffffff,den,d);
   x=0;y=0;
   #pragma unroll
   for(int w=0;w<W;w++) {
     float sc=__shfl_sync(0xffffffff,weight,w);
     x=fmaf(sc,partial[w][2*lane],x);y=fmaf(sc,partial[w][2*lane+1],y);
   }
''' +s[b:]
CUDA=s
MOD=load_inline(name='muse_reduce_cooperative',cpp_sources='''
#include <torch/extension.h>
void combine_full(torch::Tensor qkv,torch::Tensor pa,torch::Tensor pm,torch::Tensor vs,torch::Tensor out,int hq,int hkv,int length,int splits,bool pdl);
''',cuda_sources=CUDA,functions=['combine_full'],extra_cuda_cflags=['-gencode=arch=compute_90a,code=sm_90a','-O3','--use_fast_math'],extra_cflags=['-O3'],verbose=False)

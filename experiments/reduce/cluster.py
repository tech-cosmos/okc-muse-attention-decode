"""Cluster DSM variant: distribute one output head's split reduction across SMs."""
import native
from torch.utils.cpp_extension import load_inline
s=native.CUDA
s=s.replace('#include <cuda_bf16.h>','#include <cuda_bf16.h>\n#include <cooperative_groups.h>')
s=s.replace('__launch_bounds__(W*32)','__launch_bounds__(32)')
s=s.replace('unsigned lane=threadIdx.x%32,warp=threadIdx.x/32;\n int row=blockIdx.x;', '''auto cluster=cooperative_groups::this_cluster();
 unsigned lane=threadIdx.x%32,warp=cluster.block_rank();
 int row=blockIdx.x/W;''')
a=s.index(' __shared__ float partial[W][68];');b=s.index(' x=x/den*vscale;',a)
s=s[:a]+r'''
 __shared__ float partial[68];
 partial[2*lane]=x;partial[2*lane+1]=y;
 if(lane==0){partial[64]=m;partial[65]=den;}
 cluster.sync();
 if(warp==0) {
   float gm=-1e30f;
   #pragma unroll
   for(int w=0;w<W;w++)gm=fmaxf(gm,cluster.map_shared_rank(partial,w)[64]);
   x=0;y=0;den=0;
   #pragma unroll
   for(int w=0;w<W;w++) {
     float* pp=cluster.map_shared_rank(partial,w);
     float sc=exp2f(pp[64]-gm);
     x=fmaf(sc,pp[2*lane],x);y=fmaf(sc,pp[2*lane+1],y);
     den=fmaf(sc,pp[65],den);
   }
''' + s[b:]
s=s.replace('=__floats2bfloat162_rn(x,y);\n}', '=__floats2bfloat162_rn(x,y);\n }\n cluster.sync();\n}')
s=s.replace('config.gridDim=dim3(b*length*hq);config.blockDim=dim3(warps*32);','config.gridDim=dim3(b*length*hq*warps);config.blockDim=dim3(32);')
s=s.replace('cudaLaunchAttribute attr={};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;attr.val.programmaticStreamSerializationAllowed=pdl;\n config.attrs=&attr;config.numAttrs=1;', '''cudaLaunchAttribute attrs[2]={};attrs[0].id=cudaLaunchAttributeProgrammaticStreamSerialization;attrs[0].val.programmaticStreamSerializationAllowed=pdl;
 attrs[1].id=cudaLaunchAttributeClusterDimension;attrs[1].val.clusterDim.x=warps;attrs[1].val.clusterDim.y=1;attrs[1].val.clusterDim.z=1;
 config.attrs=attrs;config.numAttrs=2;''')
CUDA=s
MOD=load_inline(name='muse_reduce_cluster',cpp_sources='''
#include <torch/extension.h>
void reduce(torch::Tensor qkv,torch::Tensor pa,torch::Tensor pm,torch::Tensor vs,torch::Tensor out,int hq,int hkv,int length,int splits,int warps,bool pdl);
''',cuda_sources=CUDA,functions=['reduce'],extra_cuda_cflags=['-gencode=arch=compute_90a,code=sm_90a','-O3','--use_fast_math','-Xptxas=-v'],extra_cflags=['-O3'],verbose=False)
class Wrapper:
    def __init__(self,blocks): self.warps=blocks
    def __getitem__(self,grid):
        def run(qkv,pa,pm,vs,out,**kw):
            MOD.reduce(qkv,pa,pm,vs,out,kw['HQ'],kw['HKV'],kw['L'],kw['S'],self.warps,kw['PDL'])
        return run

"""Same hierarchy with shape-specialized integer indexing."""
import native
from torch.utils.cpp_extension import load_inline
s=native.CUDA
s=s.replace('template<int W,int N,bool PDL>','template<int W,int N,bool PDL,int HQT,int LT>')
s=s.replace('int HQ,int HKV,int L,int B,int S) {','int hqr,int hkvr,int lr,int B,int S) {\n constexpr int HQ=HQT,HKV=HQT/8,L=LT;')
s=s.replace('void reduce(at::Tensor','template<int HQT,int LT>\nvoid reduce_impl(at::Tensor')
s=s.replace('reduce_kernel<W,N,P>','reduce_kernel<W,N,P,HQT,LT>')
s=s.replace('else if(splits<=W*64){PP(W,64);}else {PP(W,128);}','else {TORCH_CHECK(false,"unsupported plan");}')
s+='''
void reduce(at::Tensor qkv,at::Tensor pa,at::Tensor pm,at::Tensor vs,at::Tensor out,int hq,int hkv,int length,int splits,int warps,bool pdl) {
 TORCH_CHECK(hq/hkv==8 && (hq==16 || hq==128) && (length==1 || length==8),"unsupported shape");
 #define CALL(H,L) reduce_impl<H,L>(qkv,pa,pm,vs,out,hq,hkv,length,splits,warps,pdl)
 if(hq==16){if(length==1){CALL(16,1);}else{CALL(16,8);}}
 else{if(length==1){CALL(128,1);}else{CALL(128,8);}}
}
'''
CUDA=s
MOD=load_inline(name='muse_reduce_specialized',cpp_sources='''
#include <torch/extension.h>
void reduce(torch::Tensor qkv,torch::Tensor pa,torch::Tensor pm,torch::Tensor vs,torch::Tensor out,int hq,int hkv,int length,int splits,int warps,bool pdl);
''',cuda_sources=CUDA,functions=['reduce'],extra_cuda_cflags=['-gencode=arch=compute_90a,code=sm_90a','-O3','--use_fast_math','-Xptxas=-v'],extra_cflags=['-O3'],verbose=False)
class Wrapper:
    def __init__(self,warps):self.warps=warps
    def __getitem__(self,grid):
        def run(qkv,pa,pm,vs,out,**kw):
            MOD.reduce(qkv,pa,pm,vs,out,kw['HQ'],kw['HKV'],kw['L'],kw['S'],self.warps,kw['PDL'])
        return run

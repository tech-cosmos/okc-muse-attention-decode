from pathlib import Path
import torch,triton,submission
from torch.utils.cpp_extension import load_inline
ROOT=Path(__file__).resolve().parent
ext=load_inline(name='muse_completion_prefetched_v1',cpp_sources='''#include <torch/extension.h>
void attention(torch::Tensor qkv,torch::Tensor cs,torch::Tensor kc,torch::Tensor vc,torch::Tensor pt,torch::Tensor seq,torch::Tensor loc,torch::Tensor ks,torch::Tensor vs,torch::Tensor pa,torch::Tensor pm,torch::Tensor counts,torch::Tensor output,int page,double scale,bool fuse);''',cuda_sources=(ROOT/'prefetched.cu').read_text(),functions=['attention'],extra_cuda_cflags=['-gencode=arch=compute_90a,code=sm_90a','-O3','--use_fast_math','-lineinfo','-Xptxas=-v'],extra_cflags=['-O3'],verbose=True)
counts={}
def kernel_fn(data,fuse=True):
 cfg=data['config'];b,L=cfg['batch'],cfg['q_len'];hq,hkv,d=cfg['num_q_heads'],cfg['num_kv_heads'],cfg['head_dim']
 if not(cfg['window']==2048 and L==1 and hq//hkv==8 and d==64 and cfg['page_size'] in(1,64) and cfg['rope_theta']):return submission.kernel_fn(data)
 qkv=data['qkv'];dev=qkv.device;S=min(32,triton.cdiv(data['page_table'].shape[1]*cfg['page_size'],64))
 key=(dev,b*hkv)
 if key not in counts:counts[key]=torch.zeros(b*hkv,device=dev,dtype=torch.int32)
 pa=torch.empty((b,hkv,S,8,64),dtype=torch.float32,device=dev);pm=torch.empty((b,hkv,S,8,2),dtype=torch.float32,device=dev);out=torch.empty((b,1,hq*64),dtype=torch.bfloat16,device=dev)
 ext.attention(qkv,submission._cos_sin(cfg['rope_theta'],64,dev),data['k_cache'],data['v_cache'],data['page_table'],data['seq_lens'],data['cache_loc'],data['k_scale'],data['v_scale'],pa,pm,counts[key],out,cfg['page_size'],cfg['sm_scale']*cfg['qk_scale_factor']*submission.LOG2E,fuse)
 if not fuse:submission._combine_hb[(b,hq//2)](qkv,pa,pm,data['v_scale'],out,HQ=hq,HKV=hkv,G=8,L=1,M=8,D=64,S=S,S_PAD=triton.next_power_of_2(S),SB=triton.next_power_of_2(S),HB=2,PDL=True,num_warps=1,launch_pdl=True)
 return out

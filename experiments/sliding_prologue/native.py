from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline
ROOT=Path(__file__).resolve().parent
CPP='''#include <torch/extension.h>
void attention(torch::Tensor qkv,torch::Tensor cs,torch::Tensor kc,torch::Tensor vc,torch::Tensor pt,torch::Tensor seq,torch::Tensor loc,torch::Tensor ks,torch::Tensor vs,torch::Tensor pa,torch::Tensor pm,int page,double scale,bool early);
'''
BUILD=ROOT/'.build';BUILD.mkdir(exist_ok=True)
module=load_inline(name='codex_sliding_prologue',build_directory=str(BUILD),cpp_sources=CPP,cuda_sources=(ROOT/'kernel.cu').read_text(),functions=['attention'],extra_cuda_cflags=['-gencode=arch=compute_90a,code=sm_90a','-O3','--use_fast_math','-lineinfo','--ptxas-options=-v'],extra_cflags=['-O3'],verbose=True)
def run(data,early=True,parts=False):
 import submission,triton
 c=data['config'];b=c['batch'];h=c['num_kv_heads'];hq=c['num_q_heads'];qkv=data['qkv'];device=qkv.device
 assert c['window']==2048 and c['q_len']==1 and hq//h==8 and c['head_dim']==64 and c['page_size'] in (1,64)
 s=min(32,triton.cdiv(data['page_table'].shape[1]*c['page_size'],64))
 pa=torch.empty((b,h,s,8,64),dtype=torch.float32,device=device);pm=torch.empty((b,h,s,8,2),dtype=torch.float32,device=device)
 cs=submission._cos_sin(c['rope_theta'],64,device)
 module.attention(qkv,cs,data['k_cache'],data['v_cache'],data['page_table'],data['seq_lens'],data['cache_loc'],data['k_scale'],data['v_scale'],pa,pm,c['page_size'],c['sm_scale']*c['qk_scale_factor']*submission.LOG2E,early)
 if parts:return pa,pm
 out=torch.empty((b,1,hq*64),dtype=torch.bfloat16,device=device)
 submission._combine_hb[(b,hq//2)](qkv,pa,pm,data['v_scale'],out,HQ=hq,HKV=h,G=8,L=1,M=8,D=64,S=s,S_PAD=triton.next_power_of_2(s),SB=triton.next_power_of_2(s),HB=2,PDL=True,num_warps=1,launch_pdl=True)
 return out

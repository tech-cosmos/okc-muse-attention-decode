"""Experimental full-decode single-key-tile attention, no production imports."""
from pathlib import Path
import importlib.util
import sys
import torch
import triton
from torch.utils.cpp_extension import load_inline
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('s28_production',ROOT/'production_snapshot.py')
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
MODULES={}
def extension(bn):
    if bn not in MODULES:
        MODULES[bn]=load_inline(name=f'muse_s28_fulltile{bn}',cpp_sources=(ROOT/'binding.cpp').read_text(),
            cuda_sources=(ROOT/f'tile{bn}.cu').read_text(),functions=['attention'],
            extra_cuda_cflags=['-gencode=arch=compute_90a,code=sm_90a','-O3','--use_fast_math','-lineinfo','--ptxas-options=-v'],
            extra_cflags=['-O3'],verbose=True)
    return MODULES[bn]

def eligible(data):
    c=data['config']
    return not c['window'] and c['q_len']==1 and c['head_dim']==64 and c['num_q_heads']//c['num_kv_heads']==8 and c['page_size'] in (1,64) and not c['rope_theta']

def kernel(data,bn=256):
    if not eligible(data):return parent.kernel_fn(data)
    c=data['config'];b=c['batch'];hq=c['num_q_heads'];hk=c['num_kv_heads'];qkv=data['qkv'];dev=qkv.device
    s=triton.cdiv(data['page_table'].shape[1]*c['page_size'],bn)
    pa=torch.empty((b,hk,s,8,64),device=dev,dtype=torch.float32)
    pm=torch.empty((b,hk,s,8,2),device=dev,dtype=torch.float32)
    out=torch.empty((b,1,hq*64),device=dev,dtype=torch.bfloat16)
    extension(bn).attention(qkv,data['k_scale'],data['k_cache'],data['v_cache'],data['page_table'],
        data['seq_lens'],data['cache_loc'],data['k_scale'],data['v_scale'],pa,pm,c['page_size'],
        c['sm_scale']*c['qk_scale_factor']*parent.LOG2E,True)
    parent._combine[(b,hq)](qkv,pa,pm,data['v_scale'],out,
        HQ=hq,HKV=hk,G=8,L=1,M=8,D=64,S=s,S_PAD=triton.next_power_of_2(s),
        SB=min(128,triton.next_power_of_2(s)),PDL=parent.USE_PDL,num_warps=4 if s>32 else 2,launch_pdl=parent.USE_PDL)
    return out

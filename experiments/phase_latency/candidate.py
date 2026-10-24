"""Offline diagnostic only. Trace buffers are fresh and returned to the caller."""
import importlib.util,sys
from pathlib import Path
import torch,triton
from torch.utils.cpp_extension import load_inline
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('phase_parent',ROOT/'production_snapshot.py')
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
ext=load_inline(name='muse_phase_latency_s32',cpp_sources=(ROOT/'binding.cpp').read_text(),cuda_sources=(ROOT/'diagnostic.cu').read_text(),functions=['attention'],extra_cuda_cflags=['-gencode=arch=compute_90a,code=sm_90a','-O3','--use_fast_math','-lineinfo'],extra_cflags=['-O3'],verbose=False)
_plan=parent._plan
PHASES=['metadata_and_gather_issue','query_newkv_preparation','residual_copy_wait','first_CTA_rendezvous','newkv_publish','value_conversion_publish','QK','softmax_pack','PV','partial_output','total','clock_pair_overhead']
def supported(cfg):
    return cfg['window']==2048 and cfg['q_len']==1 and cfg['batch']*cfg['num_kv_heads']<=16 and cfg['num_q_heads']//cfg['num_kv_heads']==8 and cfg['head_dim']==64 and cfg['page_size'] in (1,64) and bool(cfg['rope_theta'])
def run(data,phase=10):
    cfg=data['config']
    if not supported(cfg):return parent.kernel_fn(data),None
    batch,hq,hkv=cfg['batch'],cfg['num_q_heads'],cfg['num_kv_heads'];qkv=data['qkv'];device=qkv.device
    splits=min(32,triton.cdiv(data['page_table'].shape[1]*cfg['page_size'],64))
    pacc=torch.empty((batch,hkv,splits,8,64),dtype=torch.float32,device=device)
    pml=torch.empty((batch,hkv,splits,8,2),dtype=torch.float32,device=device)
    out=torch.empty((batch,1,hq*64),dtype=torch.bfloat16,device=device)
    trace=torch.empty((batch,hkv,splits,4,2),dtype=torch.int64,device=device)
    ext.attention(qkv,parent._cos_sin(cfg['rope_theta'],64,device),data['k_cache'],data['v_cache'],data['page_table'],data['seq_lens'],data['cache_loc'],data['k_scale'],data['v_scale'],pacc,pml,cfg['page_size'],cfg['sm_scale']*cfg['qk_scale_factor']*parent.LOG2E,True,trace,phase)
    parent._combine_hb[(batch,hq//2)](qkv,pacc,pml,data['v_scale'],out,HQ=hq,HKV=hkv,G=8,L=1,M=8,D=64,S=splits,S_PAD=triton.next_power_of_2(splits),SB=triton.next_power_of_2(splits),HB=2,PDL=parent.USE_PDL,num_warps=1,launch_pdl=parent.USE_PDL)
    return out,trace

def kernel_fn(data):return run(data)[0]

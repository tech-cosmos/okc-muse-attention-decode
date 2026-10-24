from pathlib import Path
import torch,os
from torch.utils.cpp_extension import load_inline
import importlib.util,sys
ROOT = Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("s36production",ROOT/"production.py")
parent=importlib.util.module_from_spec(spec);sys.modules[spec.name]=parent;spec.loader.exec_module(parent)
CPP = '''#include <torch/extension.h>
void attention(torch::Tensor qkv,torch::Tensor kc,torch::Tensor vc,torch::Tensor pt,torch::Tensor seq,torch::Tensor loc,torch::Tensor ks,torch::Tensor vs,torch::Tensor pa,torch::Tensor pm,int page,double scale);'''
VARIANT=os.environ.get('TDEC_VARIANT','deferred')
assert VARIANT in ('natural','deferred','lane')
ext = load_inline(name='muse_deferred_denominator_s36_'+VARIANT, cpp_sources=CPP,
    cuda_sources=(ROOT/({'natural':'natural_v.cu','deferred':'deferred.cu','lane':'lane.cu'}[VARIANT])).read_text(), functions=['attention'],
    extra_cuda_cflags=['-gencode=arch=compute_90a,code=sm_90a','-O3','--use_fast_math','-lineinfo'],
    extra_cflags=['-O3'], verbose=False)
_plan = parent._plan

def partials(data):
    c=data['config']; b=c['batch']; h=c['num_kv_heads']; dev=data['qkv'].device
    # One fixed native grid policy, not a tuning sweep.
    splits=min(128,max(1,(528+b*(h//2)-1)//(b*(h//2))))
    pa=torch.empty((b,h,splits,8,64),device=dev,dtype=torch.float32)
    pm=torch.empty((b,h,splits,8,2),device=dev,dtype=torch.float32)
    ext.attention(data['qkv'],data['k_cache'],data['v_cache'],data['page_table'],data['seq_lens'],data['cache_loc'],data['k_scale'],data['v_scale'],pa,pm,c['page_size'],c['sm_scale']*c['qk_scale_factor']*parent.LOG2E)
    return pa,pm

def kernel_fn(data):
    c=data['config']; b=c['batch']; h=c['num_kv_heads']; hq=c['num_q_heads']
    if c['window'] or c['q_len']!=1 or hq//h!=8 or c['head_dim']!=64 or h%2 or c['page_size'] not in (1,64) or c['rope_theta']:
        return parent.kernel_fn(data)
    pa,pm=partials(data); s=pa.shape[2]
    out=torch.empty((b,1,hq*64),device=pa.device,dtype=torch.bfloat16)
    if b*h<=4 and 64<s<=128:
        parent._NATIVE_MERGE.combine_full(data['qkv'],pa,pm,data['v_scale'],out,hq,h,1,s,True)
    else:
        parent._combine[(b,hq)](data['qkv'],pa,pm,data['v_scale'],out,HQ=hq,HKV=h,G=8,L=1,M=8,D=64,S=s,S_PAD=parent.triton.next_power_of_2(s),SB=min(128,parent.triton.next_power_of_2(s)),PDL=True,num_warps=4 if s>32 else 2,launch_pdl=True)
    return out

"""Experimental broad native L1 sliding dispatch, unchanged production fallback."""
import os
import importlib
import torch,triton,submission as parent
variant=os.environ.get('PAIR_VARIANT','base')
module=importlib.import_module('experiments.sliding_fp8pairs.'+variant).module

def eligible(data):
 c=data['config']
 return c['window']==2048 and c['q_len']==1 and c['num_q_heads']//c['num_kv_heads']==8 and c['head_dim']==64 and c['page_size'] in (1,64) and bool(c['rope_theta'])

def native(data,mod=module):
 assert eligible(data)
 c=data['config'];b,hq,hkv=c['batch'],c['num_q_heads'],c['num_kv_heads'];qkv,pt=data['qkv'],data['page_table']
 splits=min(32,triton.cdiv(pt.shape[1]*c['page_size'],64))
 pa=torch.empty((b,hkv,splits,8,64),device=qkv.device,dtype=torch.float32)
 pm=torch.empty((b,hkv,splits,8,2),device=qkv.device,dtype=torch.float32)
 out=torch.empty((b,1,hq*64),device=qkv.device,dtype=torch.bfloat16)
 cs=parent._cos_sin(c['rope_theta'],64,qkv.device)
 mod.attention(qkv,cs,data['k_cache'],data['v_cache'],pt,data['seq_lens'],data['cache_loc'],data['k_scale'],data['v_scale'],pa,pm,c['page_size'],c['sm_scale']*c['qk_scale_factor']*parent.LOG2E,True)
 parent._combine_hb[(b,hq//2)](qkv,pa,pm,data['v_scale'],out,HQ=hq,HKV=hkv,G=8,L=1,M=8,D=64,S=splits,S_PAD=triton.next_power_of_2(splits),SB=triton.next_power_of_2(splits),HB=2,PDL=parent.USE_PDL,num_warps=1,launch_pdl=parent.USE_PDL)
 return out
original=parent.kernel_fn
def kernel_fn(data):
 return native(data) if eligible(data) else original(data)

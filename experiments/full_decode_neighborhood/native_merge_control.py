"""Compose the frozen tile128 attention with the already validated native merge predicate."""
import torch,triton
from candidate import parent,eligible,extension

def kernel(data,bn=128):
    if not eligible(data):return parent.kernel_fn(data)
    c=data['config'];b=c['batch'];hq=c['num_q_heads'];hk=c['num_kv_heads'];qkv=data['qkv'];dev=qkv.device
    s=triton.cdiv(data['page_table'].shape[1]*c['page_size'],bn)
    pa=torch.empty((b,hk,s,8,64),device=dev,dtype=torch.float32)
    pm=torch.empty((b,hk,s,8,2),device=dev,dtype=torch.float32)
    out=torch.empty((b,1,hq*64),device=dev,dtype=torch.bfloat16)
    extension(bn).attention(qkv,data['k_scale'],data['k_cache'],data['v_cache'],data['page_table'],
        data['seq_lens'],data['cache_loc'],data['k_scale'],data['v_scale'],pa,pm,c['page_size'],
        c['sm_scale']*c['qk_scale_factor']*parent.LOG2E,True)
    if b*hk<=4 and 64<s<=128:
        parent._NATIVE_MERGE.combine_full(qkv,pa,pm,data['v_scale'],out,hq,hk,1,s,parent.USE_PDL)
        return out
    parent._combine[(b,hq)](qkv,pa,pm,data['v_scale'],out,
        HQ=hq,HKV=hk,G=8,L=1,M=8,D=64,S=s,S_PAD=triton.next_power_of_2(s),
        SB=min(128,triton.next_power_of_2(s)),PDL=parent.USE_PDL,num_warps=4 if s>32 else 2,launch_pdl=parent.USE_PDL)
    return out

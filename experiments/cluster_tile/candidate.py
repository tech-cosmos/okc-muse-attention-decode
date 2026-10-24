"""Offline cluster fusion experiment; no production imports this module."""
from pathlib import Path
import importlib.util,sys
import torch,triton
from torch.utils.cpp_extension import load_inline
ROOT=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('s31frozen',ROOT/'production_snapshot.py')
parent=importlib.util.module_from_spec(sp);sys.modules[sp.name]=parent;sp.loader.exec_module(parent)
mods={}
for name,src,cpp in [('cluster','cluster.cu','cluster_binding.cpp'),('unfused','tile256.cu','binding.cpp')]:
    build=ROOT/'.build'/name;build.mkdir(parents=True,exist_ok=True)
    mods[name]=load_inline(name='muse_s31_'+name,cpp_sources=(ROOT/cpp).read_text(),cuda_sources=(ROOT/src).read_text(),functions=['attention'],build_directory=str(build),extra_cuda_cflags=['-gencode=arch=compute_90a,code=sm_90a','-O3','--use_fast_math'],extra_cflags=['-O3'],verbose=False)
MODE='cluster'
def eligible(d):
    c=d['config'];return c['window']==2048 and c['q_len']==1 and c['head_dim']==64 and c['num_q_heads']//c['num_kv_heads']==8 and c['page_size'] in (1,64) and bool(c['rope_theta'])
def kernel_fn(data,mode=None):
    mode=mode or MODE
    if not eligible(data):return parent.kernel_fn(data)
    c=data['config'];q=data['qkv'];b,hq,hkv=c['batch'],c['num_q_heads'],c['num_kv_heads'];s=8
    pa=torch.empty((b,hkv,s,8,64),device=q.device,dtype=torch.float32)
    pm=torch.empty((b,hkv,s,8,2),device=q.device,dtype=torch.float32)
    out=torch.empty((b,1,hq*64),device=q.device,dtype=torch.bfloat16)
    cs=parent._cos_sin(c['rope_theta'],64,q.device)
    args=[q,cs,data['k_cache'],data['v_cache'],data['page_table'],data['seq_lens'],data['cache_loc'],data['k_scale'],data['v_scale'],pa,pm]
    if mode=='cluster':args.append(out)
    mods[mode].attention(*args,c['page_size'],c['sm_scale']*c['qk_scale_factor']*parent.LOG2E,True)
    if mode=='unfused':
        parent._combine_hb[(b,hq//2)](q,pa,pm,data['v_scale'],out,HQ=hq,HKV=hkv,G=8,L=1,M=8,D=64,S=s,S_PAD=s,SB=s,HB=2,PDL=True,num_warps=1,launch_pdl=True)
    return out
# tools/bench.py reads this hook even when no plan override is requested.
_plan=parent._plan

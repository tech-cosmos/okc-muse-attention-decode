import fcntl,sys
sys.path.insert(0,'.')
import torch,reference
import importlib
mod=importlib.import_module("experiments.sliding_conflicts."+sys.argv[1])
with open('/tmp/okc-gpu.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 d=reference.generate_input(seed=99,batch=1,ctx=80000,heads='tp8',layer='sliding',page=1,q_len=1,dist='soft')
 inv=1./(d['config']['rope_theta']**(torch.arange(0,64,2,device='cuda',dtype=torch.float32)/64))
 ang=torch.arange(80001,device='cuda')[:,None]*inv
 cs=torch.cat([ang.cos(),ang.sin()],-1).contiguous()
 pa=torch.empty((1,2,32,8,64),device='cuda');pm=torch.empty((1,2,32,8,2),device='cuda')
 for i in range(3):
  mod.module.attention(d['qkv'],cs,d['k_cache'],d['v_cache'],d['page_table'],d['seq_lens'],d['cache_loc'],d['k_scale'],d['v_scale'],pa,pm,1,d['config']['sm_scale']*d['config']['qk_scale_factor']*1.4426950408889634,sys.argv[2]=='early')
 torch.cuda.synchronize()

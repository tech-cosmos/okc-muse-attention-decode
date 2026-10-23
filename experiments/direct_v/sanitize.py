import sys,fcntl
sys.path.insert(0,'.')
import torch,reference
from experiments.direct_v import direct
with open('/tmp/okc-gpu.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 for page in (1,64):
  for scale in (1.,1./64):
   d=reference.generate_input(seed=314,batch=2,ctx=257,heads='tp8',layer='full',page=page,q_len=8,kv_scale=scale,dist='soft')
   s=4;h=2;hq=16
   pa=torch.empty((2,h,s,64,64),device='cuda');pm=torch.empty((2,h,s,64,2),device='cuda')
   for rep in range(3):
    d['qkv'].normal_()
    direct.module.attention(d['qkv'],d['k_cache'],d['v_cache'],d['page_table'],d['seq_lens'],d['cache_loc'],d['k_scale'],d['v_scale'],pa,pm,s,page,d['config']['sm_scale']*d['config']['qk_scale_factor']*1.4426950408889634,4)
    torch.cuda.synchronize()
    assert torch.isfinite(pa).all() and torch.isfinite(pm).all()
   print('PASS native-only',page,scale,flush=True)

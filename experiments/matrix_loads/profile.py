"""One native attention launch under NCU, protected by shared GPU flock."""
import os,sys,fcntl,importlib
sys.path.insert(0,'.')
import torch,reference
variant=importlib.import_module('experiments.matrix_loads.'+os.environ.get('VARIANT','baseline'))
with open('/tmp/okc-gpu.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 d=reference.generate_input(seed=99,batch=16,ctx=80000,heads='tp8',layer='full',page=1,q_len=8,spread='agentic',dist='coherent')
 s=16
 pa=torch.empty((16,2,s,64,64),device='cuda');pm=torch.empty((16,2,s,64,2),device='cuda')
 args=[d['qkv'],d['k_cache'],d['v_cache'],d['page_table'],d['seq_lens'],d['cache_loc'],d['k_scale'],d['v_scale'],pa,pm,s,1,d['config']['sm_scale']*d['config']['qk_scale_factor']*1.4426950408889634,4]
 for _ in range(3):variant.module.attention(*args)
 torch.cuda.synchronize()

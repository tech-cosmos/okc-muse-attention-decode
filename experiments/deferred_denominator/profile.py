import sys,os,fcntl
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch,reference
from experiments.deferred_denominator import native
assert os.environ.get('S36_PROFILE_LOCKED')=='1','parent must own the shared GPU lock'
d=reference.generate_input(batch=16,ctx=80000,heads='tp8',layer='full',page=1,q_len=1,spread='agentic',dist='coherent',seed=36310)
flush=torch.empty(512<<20,device='cuda',dtype=torch.uint8);flush.zero_();torch.cuda.synchronize()
o=native.kernel_fn(d);torch.cuda.synchronize();ok,msg=reference.check_implementation(d,reference.collect(d,o));assert ok,msg
print('PASS profile',native.VARIANT,msg,flush=True)

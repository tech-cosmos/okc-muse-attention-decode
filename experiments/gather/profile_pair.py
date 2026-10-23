import sys,fcntl
sys.path.insert(0,'.')
import torch,reference
from experiments.gather import pair_p4 as pair
with open('/tmp/okc-gpu.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 data=reference.generate_input(batch=16,ctx=80000,heads='tp8',layer='full',page=1,q_len=8,dist='coherent',spread='agentic',seed=99)
 for _ in range(3):pair.run(data,splits=16,stages=3)
 torch.cuda.synchronize()

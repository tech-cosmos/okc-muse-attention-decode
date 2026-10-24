"""Unmodified production attention profiling, independent of trace kernels."""
import fcntl,sys
sys.path.insert(0,'.')
import torch,reference
from experiments.phase_latency.candidate import parent
batch,page=int(sys.argv[1]),int(sys.argv[2])
with open('/tmp/okc-gpu.lock','w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    d=reference.generate_input(seed=32001,batch=batch,ctx=80000,heads='tp8',layer='sliding',page=page,q_len=1,dist='soft',spread='fixed' if batch==1 else 'agentic')
    for _ in range(3):out=parent.kernel_fn(d)
    torch.cuda.synchronize();flush=torch.empty(512<<20,device='cuda',dtype=torch.uint8);flush.zero_();torch.cuda.synchronize()
    out=parent.kernel_fn(d);torch.cuda.synchronize();ok,msg=reference.check_implementation(d,reference.collect(d,out));assert ok,msg
    print('PROFILE REFERENCE PASS',batch,page,flush=True)

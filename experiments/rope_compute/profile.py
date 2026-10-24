"""One native attention launch after cold-L2 flush; invoked only after timings."""
import fcntl,sys,os,contextlib
import run
import torch,reference
name,batch,page=sys.argv[1],int(sys.argv[2]),int(sys.argv[3])
with (contextlib.nullcontext() if os.environ.get('PROFILE_PARENT_LOCK')=='1' else open('/tmp/okc-gpu.lock','w')) as lock:
    if lock is not None:fcntl.flock(lock,fcntl.LOCK_EX)
    d=reference.generate_input(seed=34103,batch=batch,ctx=80000,heads='tp8',layer='sliding',page=page,q_len=1,dist='soft',spread='fixed' if batch==1 else 'agentic')
    m=run.MODULES[name]
    # Warm/capture companion merge, then profile the fourth native invocation.
    for _ in range(3):out=m.kernel_fn(d)
    torch.cuda.synchronize()
    flush=torch.empty(512<<20,device='cuda',dtype=torch.uint8);flush.zero_();torch.cuda.synchronize()
    out=m.kernel_fn(d)
    run.check(d,out,('profile',name,batch,page))

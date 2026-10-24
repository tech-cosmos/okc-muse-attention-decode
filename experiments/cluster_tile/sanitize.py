"""Isolated native cluster checks, with no Triton kernels launched."""
import fcntl,sys
from pathlib import Path
sys.path.insert(0,'.');sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch,reference,candidate
from run import capture,check
with open('/tmp/okc-gpu.lock','w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    count=0
    for page in (1,64):
        for scale in (1,1/64):
            for ctx in (10,257,4096):
                c=dict(batch=2,ctx=ctx,heads='tp8',layer='sliding',q_len=1,page=page,kv_scale=scale,dist='edges',spread='mixed')
                d=reference.generate_input(seed=93,**c);g,out=capture(lambda:candidate.kernel_fn(d,'cluster'))
                for i in range(2):
                    new=reference.generate_input(seed=31015+i,**c)
                    for k,v in d.items():
                        if isinstance(v,torch.Tensor):v.copy_(new[k])
                    if i==1:
                        d['seq_lens'].zero_();d['positions'].zero_();d['cache_loc'].copy_(d['page_table'][:,0].long()*page);d['cache_seqlens'].fill_(1)
                    g.replay();check(d,out,(page,scale,ctx,i));count+=1
                del d,new,g,out;torch.cuda.empty_cache()
    print('NATIVE_CLUSTER_REPLAYS',count)

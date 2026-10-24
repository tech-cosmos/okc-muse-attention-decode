"""Complete outputs with alternate config-only RoPE frequencies."""
import fcntl,json,sys
from pathlib import Path
sys.path.insert(0,'.')
import torch,reference
from experiments.rope_compute.run import candidate,check,capture,IMPLEMENTATION
records=[]
with open('/tmp/okc-gpu.lock','w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    for theta in (10000.,1000000.):
        for page in (1,64):
            c=dict(batch=2,ctx=4096,heads='tp8',layer='sliding',page=page,q_len=1,dist='soft',kv_scale=1/64)
            d=reference.generate_input(seed=34091,**c);d['config']['rope_theta']=theta
            check(d,candidate.kernel_fn(d),(theta,page,'eager'))
            g,out=capture(lambda:candidate.kernel_fn(d))
            for seed in (34092,34093):
                new=reference.generate_input(seed=seed,**c)
                for k,v in d.items():
                    if isinstance(v,torch.Tensor):v.copy_(new[k])
                g.replay();check(d,out,(theta,page,seed))
            records.append(dict(theta=theta,page=page,eager=1,replays=2))
    (Path(__file__).parent/'results'/f'{IMPLEMENTATION}-config.json').write_text(json.dumps(records,indent=2)+'\n')

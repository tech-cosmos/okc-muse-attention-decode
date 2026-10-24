"""Exhaustive configured position-domain comparison against FP32 torch sin/cos."""
import fcntl,json,sys
from pathlib import Path
sys.path.insert(0,'.')
import torch
from experiments.rope_compute.native import module,variant
ROOT=Path(__file__).resolve().parent
records=[]
with open('/tmp/okc-gpu.lock','w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    for theta in (500000.,10000.,1000000.):
        freq=1./(theta**(torch.arange(0,64,2,dtype=torch.float32,device='cuda')/64))
        count=32768
        out=torch.empty((count,32,2),device='cuda')
        maximum=torch.zeros(2,device='cuda');sums=torch.zeros(2,device='cuda',dtype=torch.float64)
        for start in range(0,2**21,count):
            module.rope_probe(freq,out,start,count)
            ang=torch.arange(start,start+count,device='cuda',dtype=torch.float32)[:,None]*freq
            ref=torch.stack((ang.cos(),ang.sin()),dim=-1)
            diff=(out-ref).abs()
            maximum=torch.maximum(maximum,diff.amax((0,1)))
            sums+=diff.double().square().sum((0,1))
            assert torch.isfinite(out).all()
        rec=dict(theta=theta,positions=2**21,angles=2**21*32,max_abs=maximum.tolist(),rms=(sums/(2**21*32)).sqrt().tolist())
        assert max(rec['max_abs'])<1.e-6,rec
        records.append(rec);print(json.dumps(rec),flush=True)
    (ROOT/'results'/f'{variant}-accuracy.json').write_text(json.dumps(records,indent=2)+'\n')

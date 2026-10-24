"""Isolated native merge graph tests; no Triton loaders execute under sanitizer."""
import ast, fcntl, json, sys
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline
ROOT=Path(__file__).resolve().parent
source=ast.parse((ROOT/'candidate.py').read_text())
node=next(n for n in source.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_NATIVE_MERGE' for t in n.targets))
ns={'load_inline':load_inline}
exec(compile(ast.Module(body=[node],type_ignores=[]),'native-extract','exec'),ns)
merge=ns['_NATIVE_MERGE']

def run():
    records=[]
    for dtype in (torch.float32,torch.bfloat16):
        for pdl in (False,True):
            for s,length,b,hkv in ((1,1,1,2),(16,8,1,2),(64,1,2,2),(65,8,2,2),(76,8,1,16),(128,8,1,2)):
                hq=hkv*8;rows=length*8;width=(2*hq+2*hkv)*64
                q=torch.empty((b,length,width),device='cuda',dtype=torch.bfloat16)
                pa=torch.empty((b,hkv,s,rows,64),device='cuda',dtype=dtype)
                pm=torch.empty((b,hkv,s,rows,2),device='cuda',dtype=torch.float32)
                vs=torch.ones(1,device='cuda');out=torch.empty((b,length,hq*64),device='cuda',dtype=torch.bfloat16)
                def fill(rep):
                    torch.manual_seed(370000+rep+s)
                    q.normal_();pa.normal_();pa.mul_(100000 if rep==2 else 1)
                    pm[...,0].uniform_(-70,0);pm[...,1].uniform_(1,2000)
                    if s>1:
                        # Empty partials, including whole empty warp groups.
                        pa[:,:,1::2].zero_();pm[:,:,1::2, :,0].fill_(-1e30);pm[:,:,1::2,:,1].zero_()
                    vs.fill_(1/64 if rep==1 else 1)
                fill(0)
                def launch():merge.combine_full(q,pa,pm,vs,out,hq,hkv,length,s,pdl)
                launch();torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):launch()
                errors=[]
                for rep in range(3):
                    fill(rep);graph.replay();torch.cuda.synchronize()
                    w=torch.exp2(pm[...,0]-pm[...,0].amax(2,keepdim=True))
                    v=(pa.float()*w[...,None]).sum(2)/(pm[...,1]*w).sum(2)[...,None]*vs
                    v=v.reshape(b,hkv,length,8,64).permute(0,2,1,3,4).reshape(b,length,hq,64)
                    gate=q[..., (hq+2*hkv)*64:].float().reshape(b,length,hq,64)
                    expected=v*torch.rsqrt(v.square().mean(-1,keepdim=True)+1e-5)*torch.sigmoid(gate)
                    actual=out.float().reshape_as(expected)
                    rel=((actual-expected).square().sum()/expected.square().sum()).sqrt().item()
                    assert torch.isfinite(actual).all() and rel<0.003,(dtype,pdl,s,rep,rel)
                    torch.testing.assert_close(actual,expected,atol=.016,rtol=.012)
                    errors.append(rel)
                record=dict(dtype=str(dtype),pdl=pdl,splits=s,length=length,batch=b,hkv=hkv,replays=3,relative_l2=errors)
                records.append(record);print(json.dumps(record),flush=True)
    (ROOT/('synthetic-'+(sys.argv[1] if len(sys.argv)>1 else 'plain')+'.json')).write_text(json.dumps(records,indent=2)+'\n')
    print('PASS',len(records),'cases',sum(r['replays'] for r in records),'fresh graph replays',flush=True)
if __name__=='__main__':
    with open('/tmp/okc-gpu.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);run()

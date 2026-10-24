"""Isolated native merge sanitizer/reference driver, no Triton import."""
import fcntl,sys
sys.path.insert(0,'.')
import torch
from experiments.verify_merge.native import ext

def expected(qkv,pa,pm,vs):
 b,hkv,s,m,d=pa.shape;l=qkv.shape[1];g=m//l;hq=hkv*g
 w=(pm[...,0]-pm[...,0].amax(2,keepdim=True)).exp2()
 v=(pa*w[...,None]).sum(2)/(w*pm[...,1]).sum(2)[...,None]*vs
 v=v.reshape(b,hkv,l,g,64).permute(0,2,1,3,4).reshape(b,l,hq,64)
 v=v*torch.rsqrt(v.square().mean(-1,keepdim=True)+1e-5)
 gate=qkv[...,(hq+2*hkv)*64:].float().reshape(b,l,hq,64)
 return (v*gate.sigmoid()).to(torch.bfloat16).reshape(b,l,hq*64)

with open('/tmp/okc-gpu.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 count=0
 for striped in (False,True):
  for s,hkv,scale in [(s,2,1.) for s in (5,9,17,22,32,33,64)]+[(17,16,1/64)]:
   b,l,g=1,8,8;hq=hkv*g
   pa=torch.empty((b,hkv,s,l*g,64),device='cuda');pm=torch.empty((b,hkv,s,l*g,2),device='cuda')
   qkv=torch.empty((b,l,(2*hq+2*hkv)*64),device='cuda',dtype=torch.bfloat16)
   out=torch.empty((b,l,hq*64),device='cuda',dtype=torch.bfloat16);vs=torch.tensor([scale],device='cuda')
   pa.normal_();pm[...,0].normal_();pm[...,1].uniform_(.25,3);qkv.normal_()
   ext.merge(qkv,pa,pm,vs,out,hq,hkv,l,s,striped,True);torch.cuda.synchronize()
   graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):ext.merge(qkv,pa,pm,vs,out,hq,hkv,l,s,striped,True)
   for seed in (11,22,33):
    torch.manual_seed(seed);pa.normal_();pm[...,0].uniform_(-30,30);pm[...,1].uniform_(.25,3);qkv.normal_()
    # Legal empty partials: zero vectors/denominator and -1e30 max.
    pa[:,:,0].zero_();pm[:,:,0,:,0].fill_(-1e30);pm[:,:,0,:,1].zero_()
    graph.replay();torch.cuda.synchronize();ref=expected(qkv,pa,pm,vs)
    err=(out.float()-ref.float()).norm()/ref.float().norm()
    assert torch.isfinite(out).all() and err<.006,(striped,s,hkv,seed,err.item())
    count+=1
   print('PASS synthetic',striped,s,hkv,scale,'3 fresh graph replays',flush=True)
   del graph,pa,pm,qkv,out,vs,ref;torch.cuda.empty_cache()
 print('PASS total',count,flush=True)

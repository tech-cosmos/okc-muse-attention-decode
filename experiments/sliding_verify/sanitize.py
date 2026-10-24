"""Only the native kernel and torch reference; no Triton loader probes."""
import fcntl,sys
sys.path.insert(0,'.')
import torch,reference
from experiments.sliding_verify import native
with open('/tmp/okc-gpu.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 for early in (False,True):
  for page in (1,64):
   for scale in (1.,1./64):
    b,h,hq,ctx=2,2,16,2053
    d=reference.generate_input(seed=314,batch=b,ctx=ctx,heads='tp8',layer='sliding',page=page,q_len=8,kv_scale=scale,dist='edges')
    s=33
    pa=torch.empty((b,h,s,64,64),device='cuda');pm=torch.empty((b,h,s,64,2),device='cuda')
    inv=1./(d['config']['rope_theta']**(torch.arange(0,64,2,device='cuda',dtype=torch.float32)/64))
    ang=torch.arange(ctx+8,device='cuda')[:,None]*inv
    cs=torch.cat([ang.cos(),ang.sin()],-1).contiguous()
    for rep in range(3):
     d['qkv'].normal_()
     native.module.attention(d['qkv'],cs,d['k_cache'],d['v_cache'],d['page_table'],d['seq_lens'],d['cache_loc'],d['k_scale'],d['v_scale'],pa,pm,page,d['config']['sm_scale']*d['config']['qk_scale_factor']*1.4426950408889634,early)
     torch.cuda.synchronize();assert torch.isfinite(pa).all() and torch.isfinite(pm).all()
     w=torch.exp2(pm[...,0]-pm[...,0].amax(2,keepdim=True))
     a=((pa*w[...,None]).sum(2)/(w*pm[...,1]).sum(2)[...,None]*d['v_scale']).reshape(b,h,8,8,64).permute(0,2,1,3,4).reshape(b,8,hq,64)
     gate=d['qkv'][...,(hq+2*h)*64:].float().reshape_as(a)
     out=(a*torch.rsqrt(a.square().mean(-1,keepdim=True)+1e-5)*torch.sigmoid(gate)).flatten(2).bfloat16()
     ok,msg=reference.check_implementation(d,reference.collect(d,out));assert ok,(early,page,scale,msg)
    print('PASS native-only',early,page,scale,flush=True)

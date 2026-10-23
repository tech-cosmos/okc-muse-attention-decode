"""Isolated native kernels: no Triton loading, fresh values and an independent partial-output oracle."""
import sys,os,fcntl
sys.path.insert(0,'.')
import torch,reference
from experiments.key_layout import rawvk64, paired128
with open('/tmp/okc-gpu.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 for variant in (rawvk64,paired128):
  for page in (1,64):
   for scale in (1.,1./64):
    d=reference.generate_input(seed=314,batch=2,ctx=257,heads='tp8',layer='full',page=page,q_len=8,kv_scale=scale,dist='soft')
    s=4;h=2;hq=16
    pa=torch.empty((2,h,s,64,64),device='cuda');pm=torch.empty((2,h,s,64,2),device='cuda')
    for rep in range(3):
     d['qkv'].normal_()
     variant.module.attention(d['qkv'],d['k_cache'],d['v_cache'],d['page_table'],d['seq_lens'],d['cache_loc'],d['k_scale'],d['v_scale'],pa,pm,s,page,d['config']['sm_scale']*d['config']['qk_scale_factor']*1.4426950408889634,4)
     torch.cuda.synchronize()
     assert torch.isfinite(pa).all() and torch.isfinite(pm).all()
     # Reduce the native FP32 partials in torch, then independently check against reference.
     weight=torch.exp2(pm[...,0]-pm[...,0].amax(2,keepdim=True))
     a=(pa*weight[...,None]).sum(2)/(weight*pm[...,1]).sum(2)[...,None]
     a=(a*d['v_scale']).reshape(2,h,8,8,64).permute(0,2,1,3,4).reshape(2,8,hq,64)
     gate=d['qkv'][...,(hq+2*h)*64:].float().reshape_as(a)
     out=(a*torch.rsqrt(a.square().mean(-1,keepdim=True)+1e-5)*torch.sigmoid(gate)).flatten(2).bfloat16()
     ok,msg=reference.check_implementation(d,reference.collect(d,out));assert ok,msg
    print('PASS native-only reference',variant.__name__,page,scale,flush=True)

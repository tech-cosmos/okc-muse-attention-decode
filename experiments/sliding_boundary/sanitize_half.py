import fcntl,sys
sys.path.insert(0,'.')
import torch,reference
from experiments.sliding_boundary import half as pruned
with open('/tmp/okc-gpu.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 for mod in (pruned,):
  for early in (True,):
   for page in (1,64):
    for scale in (1.,1./64):
     b,h,hq,ctx=6,2,16,2048
     d=reference.generate_input(seed=314,batch=b,ctx=ctx,heads='tp8',layer='sliding',page=page,q_len=1,kv_scale=scale,dist='soft')
     s=min(32,(ctx+1+63)//64)
     pa=torch.empty((b,h,s,8,64),device='cuda');pm=torch.empty((b,h,s,8,2),device='cuda')
     inv=1./(d['config']['rope_theta']**(torch.arange(0,64,2,device='cuda',dtype=torch.float32)/64))
     ang=torch.arange(ctx+1,device='cuda')[:,None]*inv
     cs=torch.cat([ang.cos(),ang.sin()],-1).contiguous()
     for rep in range(3):
      d['qkv'].normal_()
      mod.module.attention(d['qkv'],cs,d['k_cache'],d['v_cache'],d['page_table'],d['seq_lens'],d['cache_loc'],d['k_scale'],d['v_scale'],pa,pm,page,d['config']['sm_scale']*d['config']['qk_scale_factor']*1.4426950408889634,early)
      torch.cuda.synchronize();assert torch.isfinite(pa).all() and torch.isfinite(pm).all()
      w=torch.exp2(pm[...,0]-pm[...,0].amax(2,keepdim=True))
      a=((pa*w[...,None]).sum(2)/(w*pm[...,1]).sum(2)[...,None]*d['v_scale']).reshape(b,1,hq,64)
      gate=d['qkv'][...,(hq+2*h)*64:].float().reshape_as(a)
      out=(a*torch.rsqrt(a.square().mean(-1,keepdim=True)+1e-5)*torch.sigmoid(gate)).flatten(2).bfloat16()
      ok,msg=reference.check_implementation(d,reference.collect(d,out));assert ok,(mod.__name__,early,page,scale,msg)
     print('PASS native-only',mod.__name__,early,page,scale,flush=True)

import fcntl,sys
sys.path.insert(0,'.')
import torch,reference
from experiments.sliding_completion import prefetched as native
with open('/tmp/okc-gpu.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 for ctx in(257,2048):
  for page in(1,64):
   for scale in(1.,1./64):
    b,h,hq=2,2,16
    d=reference.generate_input(seed=314,batch=b,ctx=ctx,heads='tp8',layer='sliding',page=page,q_len=1,kv_scale=scale,dist='soft')
    s=min(32,(d['page_table'].shape[1]*page+63)//64)
    pa=torch.empty((b,h,s,8,64),device='cuda');pm=torch.empty((b,h,s,8,2),device='cuda')
    output=torch.empty((b,1,hq*64),device='cuda',dtype=torch.bfloat16);counts=torch.zeros(b*h,device='cuda',dtype=torch.int32)
    inv=1./(d['config']['rope_theta']**(torch.arange(0,64,2,device='cuda',dtype=torch.float32)/64))
    ang=torch.arange(ctx+1,device='cuda')[:,None]*inv;cs=torch.cat([ang.cos(),ang.sin()],-1).contiguous()
    for rep in range(3):
     d['qkv'].normal_()
     native.ext.attention(d['qkv'],cs,d['k_cache'],d['v_cache'],d['page_table'],d['seq_lens'],d['cache_loc'],d['k_scale'],d['v_scale'],pa,pm,counts,output,page,d['config']['sm_scale']*d['config']['qk_scale_factor']*1.4426950408889634,True)
     torch.cuda.synchronize();assert not torch.count_nonzero(counts).item()
     ok,msg=reference.check_implementation(d,reference.collect(d,output));assert ok,(ctx,page,scale,rep,msg)
    print('PASS native-only fused',ctx,page,scale,'three fresh QKV calls',flush=True)

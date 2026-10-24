"""Native attention only; independently reduce FP32 partials with Torch."""
import fcntl,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch,reference
from experiments.transposed_decode import native

def run():
 count=0
 for page in (1,64):
  for scale in (1.,1/64):
   for ctx in (10,257,4096):
    kwargs=dict(batch=2,ctx=ctx,heads='tp8',layer='full',page=page,q_len=1,kv_scale=scale)
    d=reference.generate_input(seed=35200,**kwargs);c=d['config']
    pa,pm=native.partials(d);s=pa.shape[2]
    def call():
     native.ext.attention(d['qkv'],d['k_cache'],d['v_cache'],d['page_table'],d['seq_lens'],d['cache_loc'],d['k_scale'],d['v_scale'],pa,pm,page,c['sm_scale']*c['qk_scale_factor']*native.parent.LOG2E)
    call();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):call()
    for rep in range(2):
     n=reference.generate_input(seed=35201+rep,**kwargs)
     for k,v in d.items():
      if isinstance(v,torch.Tensor):v.copy_(n[k])
     if rep==1:
      d['seq_lens'].zero_();d['positions'].zero_();d['cache_loc'].copy_(d['page_table'][:,0].long()*page);d['cache_seqlens'].fill_(1)
     g.replay();torch.cuda.synchronize()
     m=pm[...,0].amax(2,keepdim=True);w=torch.exp2(pm[...,0]-m);den=(w*pm[...,1]).sum(2)
     a=(pa*w[...,None]).sum(2)/den[...,None]*d['v_scale'];a=a.reshape(2,16,64)
     a*=torch.rsqrt((a*a).mean(-1,keepdim=True)+1e-5)
     gate=d['qkv'][:,0,(16+4)*64:].float().reshape(2,16,64)
     out=(a*torch.sigmoid(gate)).to(torch.bfloat16).reshape(2,1,1024)
     ok,msg=reference.check_implementation(d,reference.collect(d,out));assert ok,msg
     print('PASS',native.VARIANT,page,scale,ctx,rep,msg,flush=True);count+=1
 print('TOTAL',count,flush=True)
if __name__=='__main__':
 with open('/tmp/okc-gpu.lock','w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);run()

import fcntl,sys
sys.path.insert(0,'.')
import torch,reference,submission
from experiments.sliding_completion import prefetched as native
import types
integrated=types.SimpleNamespace(__name__="integrated",run=lambda data,early=True: native.kernel_fn(data))
with open('/tmp/okc-gpu.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 for mod in (integrated,):
  for early in (True,):
   for b,ctx,page,heads,scale,dist in [(1,10,1,'tp8',1.,'soft'),(2,1024,64,'tp8',1./64,'edges'),(1,2047,1,'tp8',1.,'coherent'),(2,2048,64,'tp8',1.,'soft'),(2,4000,1,'tp8',1.,'edges'),(1,80000,1,'tp8',1.,'soft'),(1,2049,64,'dp',1./64,'soft'),(2,120000,64,'tp8',1./64,'coherent')]:
    c=dict(batch=b,ctx=ctx,heads=heads,layer='sliding',page=page,q_len=1,kv_scale=scale,dist=dist)
    data=reference.generate_input(seed=101,**c);mod.run(data,early=early);torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=mod.run(data,early=early)
    for i in range(6):
     fresh=reference.generate_input(seed=200+i,**c)
     for k,v in data.items():
      if isinstance(v,torch.Tensor):
       assert v.shape==fresh[k].shape,(k,v.shape,fresh[k].shape)
       v.copy_(fresh[k])
     data['seq_lens'].copy_(torch.randint(0,ctx+1,(b,),device='cuda',dtype=torch.int32))
     if i==5:data['seq_lens'].zero_()
     pos=data['seq_lens'].long()[:,None]
     data['positions'].copy_(pos.flatten())
     data['cache_loc'].copy_((torch.gather(data['page_table'].long(),1,pos//page)*page+pos%page).flatten())
     data['cache_seqlens'].copy_(data['seq_lens']+1)
     g.replay();torch.cuda.synchronize()
     ok,msg=reference.check_implementation(data,reference.collect(data,out));assert ok,(mod.__name__,early,c,i,msg)
    assert all(not torch.count_nonzero(c).item() for c in native.counts.values())
    print('PASS fresh graph',mod.__name__,early,c,'6 seeds, including empty contexts',flush=True)
    del data,g,out,fresh;torch.cuda.empty_cache()

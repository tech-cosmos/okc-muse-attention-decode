import fcntl,sys
sys.path.insert(0,'.')
import torch,reference,submission

import types
integrated=types.SimpleNamespace(__name__="integrated",run=lambda data,early=True: submission.kernel_fn(data))
with open('/tmp/okc-gpu.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 for mod in (integrated,):
  for early in (True,):
   for b,ctx,page,heads,scale,layer,length in [(3,4000,1,'tp8',1.,'sliding',1),(8,80000,64,'tp8',1./64,'sliding',1),(1,2049,64,'dp',1.,'sliding',1),(17,3000,64,'tp8',1.,'full',8),(24,8192,1,'tp8',1./64,'full',8),(32,80000,1,'tp8',1.,'full',8),(3,4000,64,'dp',1./64,'full',8),(16,6000,1,'tp8',1.,'full',8)]:
    c=dict(batch=b,ctx=ctx,heads=heads,layer=layer,page=page,q_len=length,kv_scale=scale,dist='coherent')
    data=reference.generate_input(seed=101,**c);mod.run(data,early=early);torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=mod.run(data,early=early)
    for i in range(3):
     fresh=reference.generate_input(seed=200+i,**c)
     for k,v in data.items():
      if isinstance(v,torch.Tensor):
       assert v.shape==fresh[k].shape,(k,v.shape,fresh[k].shape)
       v.copy_(fresh[k])
     data['seq_lens'].copy_(torch.randint(0,ctx+1,(b,),device='cuda',dtype=torch.int32))
     if i==2:data['seq_lens'].zero_()
     pos=data['seq_lens'].long()[:,None]+torch.arange(length,device='cuda')[None,:]
     data['positions'].copy_(pos.flatten())
     data['cache_loc'].copy_((torch.gather(data['page_table'].long(),1,pos//page)*page+pos%page).flatten())
     data['cache_seqlens'].copy_(data['seq_lens']+length)
     g.replay();torch.cuda.synchronize()
     ok,msg=reference.check_implementation(data,reference.collect(data,out));assert ok,(mod.__name__,early,c,i,msg)
    print('PASS fresh graph',mod.__name__,early,c,'3 seeds, including empty contexts',flush=True)
    del data,g,out,fresh;torch.cuda.empty_cache()

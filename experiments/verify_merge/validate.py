import argparse,fcntl,sys,types
sys.path.insert(0,'.')
import torch,reference,submission
from experiments.verify_merge import candidate
from tools import bench
p=argparse.ArgumentParser();p.add_argument('--variant',choices=['warp','striped'],required=True);p.add_argument('--serial',action='store_true');a=p.parse_args()
run=lambda data:candidate.run(data,a.variant=='striped',not a.serial)
bench.submission=types.SimpleNamespace(kernel_fn=run,_plan=submission._plan)
with open('/tmp/okc-gpu.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 assert bench.test()==0
 print('FULL SUITE 16/16; these standard cases all use unchanged production fallback',flush=True)
 for b,ctx,page,heads,scale,dist in [(17,4096,1,'tp8',1.,'soft'),(24,8192,64,'tp8',1/64,'edges'),(32,80000,1,'tp8',1.,'coherent'),(48,1024,64,'tp8',1.,'soft'),(64,6000,1,'tp8',1/64,'coherent'),(3,4000,64,'dp',1/64,'coherent'),(8,10000,1,'dp',1.,'edges')]:
  c=dict(batch=b,ctx=ctx,page=page,heads=heads,kv_scale=scale,dist=dist,layer='full',q_len=8)
  data=reference.generate_input(seed=101,**c)
  assert submission._plan(data['config'],data['page_table'].shape[1],data['qkv'].device).get('gluon')
  run(data);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):out=run(data)
  for i in range(6):
   fresh=reference.generate_input(seed=220+i,**c)
   for k,v in data.items():
    if isinstance(v,torch.Tensor):assert v.shape==fresh[k].shape;v.copy_(fresh[k])
   data['seq_lens'].copy_(torch.randint(0,ctx+1,(b,),device='cuda',dtype=torch.int32))
   if i==5:data['seq_lens'].zero_()
   pos=data['seq_lens'].long()[:,None]+torch.arange(8,device='cuda')[None,:]
   data['positions'].copy_(pos.flatten())
   data['cache_loc'].copy_((torch.gather(data['page_table'].long(),1,pos//page)*page+pos%page).flatten())
   data['cache_seqlens'].copy_(data['seq_lens']+8)
   g.replay();torch.cuda.synchronize();ok,msg=reference.check_implementation(data,reference.collect(data,out));assert ok,(c,i,msg)
  print('PASS native fresh',c,'6 seeds including empty context',flush=True)
  del data,fresh,g,out;torch.cuda.empty_cache()
print('42 native fresh graph replays PASS',flush=True)
bench.bench()

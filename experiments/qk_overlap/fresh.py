"""Correctness and fresh-input replay validation for the best paired-head prototype."""
import sys,os,importlib
sys.path.insert(0,'.')
import torch,reference,submission
from tools import bench
import importlib
combined=importlib.import_module('experiments.qk_overlap.'+os.environ.get('VARIANT','overlap'))
pair=combined
import fcntl
_gpu_lock=open('/tmp/okc-gpu.lock','w')
fcntl.flock(_gpu_lock,fcntl.LOCK_EX)
original=submission.kernel_fn

def candidate(data):
 c=data['config']
 if c['q_len']==8 and c['window']==0 and c['head_dim']==64 and c['num_q_heads']==c['num_kv_heads']*8 and c['num_kv_heads']%2==0 and c['page_size'] in (1,64):
  return pair.run(data,stages=4)
 return original(data)

submission.kernel_fn=candidate

for b,ctx,page,heads,scale,dist in [(1,10,1,'tp8',1.,'soft'),(2,1024,64,'tp8',1./64,'edges'),(3,4000,1,'tp8',1.,'coherent'),(4,12345,64,'tp8',1.,'soft'),(8,40000,1,'tp8',1.,'coherent'),(12,120000,1,'tp8',1.,'soft'),(1,2048,64,'dp',1./64,'coherent')]:
 c=dict(batch=b,ctx=ctx,heads=heads,layer='full',page=page,q_len=8,kv_scale=scale,dist=dist)
 data=reference.generate_input(seed=101,**c)
 candidate(data);torch.cuda.synchronize()
 g=torch.cuda.CUDAGraph();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(stream):
  with torch.cuda.graph(g,stream=stream):out=candidate(data)
 torch.cuda.current_stream().wait_stream(stream)
 for i in range(6):
  fresh=reference.generate_input(seed=200+i,**c)
  for k,v in data.items():
   if isinstance(v,torch.Tensor):
    assert v.shape==fresh[k].shape,(k,v.shape,fresh[k].shape)
    v.copy_(fresh[k])
  if True:
   data['seq_lens'].copy_(torch.randint(0,ctx+1,(b,),device='cuda',dtype=torch.int32))
   if i==5:data['seq_lens'].zero_()
   pos=data['seq_lens'].long()[:,None]+torch.arange(8,device='cuda')
   data['positions'].copy_(pos.flatten())
   data['cache_loc'].copy_((torch.gather(data['page_table'].long(),1,pos//page)*page+pos%page).flatten())
   data['cache_seqlens'].copy_(data['seq_lens']+8)
  g.replay();torch.cuda.synchronize()
  ok,msg=reference.check_implementation(data,reference.collect(data,out))
  assert ok,(c,i,msg)
 print('PASS fresh graph',c,'6 seeds including empty contexts',flush=True)
 del data,g,out,fresh;torch.cuda.empty_cache()
submission.kernel_fn=original
if os.environ.get('RUN_PRODUCTION')=='1':
 print('production test',flush=True)
 assert bench.test()==0

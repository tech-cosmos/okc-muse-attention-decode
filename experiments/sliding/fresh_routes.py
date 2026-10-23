"""Stress the native PDL merge with fresh values AND fresh routing on each replay."""
import sys,torch
sys.path.insert(0,'.')
import reference,submission
cases=[dict(batch=b,ctx=3000,heads='tp8',layer='sliding',q_len=l,page=p,kv_scale=s,spread='mixed',dist=dist)
       for b,l,p,s,dist in [(1,1,1,1,'soft'),(2,1,64,1/64,'edges'),(1,8,1,1,'edges'),(2,8,64,1/64,'soft')]]
for i,c in enumerate(cases):
 data=reference.generate_input(seed=40+i,**c)
 submission.kernel_fn(data);torch.cuda.synchronize()
 g=torch.cuda.CUDAGraph();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(stream):
  submission.kernel_fn(data)
  with torch.cuda.graph(g,stream=stream):out=submission.kernel_fn(data)
 torch.cuda.current_stream().wait_stream(stream)
 for step in range(20):
  fresh=reference.generate_input(seed=900+i*100+step,**c)
  for key,value in fresh.items():
   if isinstance(value,torch.Tensor):data[key].copy_(value)
  g.replay();torch.cuda.synchronize()
  ok,msg=reference.check_implementation(data,reference.collect(data,out))
  if not ok:raise RuntimeError((c,step,msg))
 print('PASS',c,'20 fresh routing/value replays',flush=True)

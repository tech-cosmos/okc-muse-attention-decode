import sys,torch
sys.path.insert(0,'.')
import submission,reference
for b,l,page,scale in ((1,1,1,1),(2,1,64,1/64),(1,8,1,1),(2,8,64,1/64)):
 c=dict(batch=b,ctx=2047,heads='tp8',layer='sliding',q_len=l,page=page,kv_scale=scale,dist='soft')
 data=reference.generate_input(seed=398,**c)
 for i in range(3):
  data['qkv'].normal_();out=submission.kernel_fn(data);torch.cuda.synchronize()
  ok,msg=reference.check_implementation(data,reference.collect(data,out))
  if not ok:raise RuntimeError(msg)
 print('PASS',c,flush=True)

import fcntl,sys
sys.path.insert(0,'.')
import torch,reference,submission
from tools import bench
from experiments.sliding_prologue import vector
original=submission.kernel_fn
for mod in (vector,):
 for early in (False,True):
  def candidate(data):
   c=data['config']
   if c['window']==2048 and c['q_len']==1 and c['num_q_heads']==8*c['num_kv_heads'] and c['head_dim']==64 and c['page_size'] in (1,64):return mod.run(data,early=early)
   return original(data)
  submission.kernel_fn=candidate
  print('VARIANT',mod.__name__,'early',early,flush=True)
  with open('/tmp/okc-gpu.lock','w') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   assert bench.test()==0
  bench.bench()
submission.kernel_fn=original

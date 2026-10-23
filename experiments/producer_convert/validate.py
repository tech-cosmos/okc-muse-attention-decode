import fcntl,sys,os
sys.path.insert(0,'.')
import torch,reference,submission
from tools import bench
from experiments.producer_convert import producer,async_convert
direct=producer if os.environ.get("VARIANT")=="producer" else async_convert
native_count=0

def run(d):
 global native_count
 c=d['config']
 if not c['window'] and c['q_len']==8 and c['num_q_heads']//c['num_kv_heads']==8 and c['page_size'] in (1,64):
  native_count+=1
  return direct.run(d,stages=4,**({"producers":4} if direct is producer else {"converters":2}))
 return submission.kernel_fn(d)

with open('/tmp/okc-gpu.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 for i,c in enumerate(bench.TESTS):
  d=reference.generate_input(seed=4242+i,**c);out=run(d)
  torch.cuda.synchronize();ok,msg=reference.check_implementation(d,reference.collect(d,out))
  print('PASS' if ok else 'FAIL',i,c,msg[:130],flush=True);assert ok,msg
  del d,out;torch.cuda.empty_cache()
print('suite passed; native cases',native_count,flush=True)

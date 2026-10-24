import fcntl,sys
sys.path.insert(0,'.')
import submission
from tools import bench
from experiments.sliding_verify import precomputed as native
parent=submission.kernel_fn
def run(d):
 c=d['config']
 return native.run(d,early=early) if c['window']==2048 and c['q_len']==8 else parent(d)
submission.kernel_fn=run
# bench.py owns the lock around timings; take it for correctness separately.
for early in (False,True):
 print('EARLY',early,flush=True)
 with open('/tmp/okc-gpu.lock','w') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  assert bench.test()==0
 bench.bench()

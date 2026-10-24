import fcntl,sys
sys.path.insert(0,'.')
from tools import bench
from experiments.sliding_completion import prefetched as native
with open('/tmp/okc-gpu.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 original=bench.submission.kernel_fn
 def test_fn(data):
  c=data['config']
  if c['window']==2048 and c['q_len']==1:return native.kernel_fn(data)
  return original(data)
 bench.submission.kernel_fn=test_fn
 assert bench.test()==0
 bench.submission.kernel_fn=original

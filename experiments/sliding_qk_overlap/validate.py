import sys,fcntl
sys.path.insert(0,'.')
from tools import bench
from experiments.sliding_qk_overlap import candidate
bench.submission.kernel_fn=candidate.kernel_fn
with open('/tmp/okc-gpu.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 bad=bench.test()
assert bad==0,bad

import fcntl,sys
sys.path.insert(0,'.')
from tools import bench
from experiments.sliding_boundary import candidate,pruned
original=bench.submission.kernel_fn
# Exercise native code at every valid test shape, including outside final dispatch.
def run(data):
 return pruned.native(data) if candidate.eligible(data) else original(data)
with open('/tmp/okc-gpu.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 bench.submission.kernel_fn=run
 assert bench.test()==0
 bench.submission.kernel_fn=original

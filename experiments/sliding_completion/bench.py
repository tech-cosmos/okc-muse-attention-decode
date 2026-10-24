import sys
sys.path.insert(0,'.')
from tools import bench
from experiments.sliding_completion import native,prefetched
original=bench.submission.kernel_fn
for mod in(native,prefetched):
 def call(data):
  c=data['config']
  if c['window']==2048 and c['q_len']==1:return mod.kernel_fn(data)
  return original(data)
 bench.submission.kernel_fn=call
 print('VARIANT',mod.__name__,flush=True)
 bench.bench()
 bench.submission.kernel_fn=original

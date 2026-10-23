import sys
sys.path.insert(0,'.')
from tools import bench
from experiments.direct_v import direct
base=bench.submission.kernel_fn

def candidate(d):
 c=d['config']
 if not c['window'] and c['q_len']==8 and c['num_q_heads']//c['num_kv_heads']==8 and c['page_size'] in (1,64):
  return direct.run(d,stages=4)
 return base(d)
bench.submission.kernel_fn=candidate
bench.bench()

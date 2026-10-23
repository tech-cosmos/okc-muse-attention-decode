import sys,os
sys.path.insert(0,'.')
import submission
from tools import bench
import importlib
combined=importlib.import_module('experiments.pv_warpgroup.'+os.environ.get('VARIANT','overlap'))
variant=combined
original=submission.kernel_fn
def candidate(data):
 c=data['config']
 if not c['window'] and c['q_len']==8 and c['num_q_heads']//c['num_kv_heads']==8 and c['page_size'] in (1,64):
  return variant.run(data,stages=4)
 return original(data)
submission.kernel_fn=candidate
bench.bench()

import sys,os
sys.path.insert(0,'.')
import submission
from tools import bench
from experiments.producer_convert import producer,async_convert
original=submission.kernel_fn

def candidate(d):
 c=d['config']
 if c['window']==0 and c['q_len']==8 and c['num_q_heads']//c['num_kv_heads']==8 and c['num_kv_heads']%2==0 and c['page_size'] in (1,64):
  if os.environ.get('VARIANT')=='producer':return producer.run(d,stages=4,producers=4)
  return async_convert.run(d,stages=4,converters=2)
 return original(d)
submission.kernel_fn=candidate
bench.bench()

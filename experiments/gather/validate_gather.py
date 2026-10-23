import sys,itertools,os,importlib
sys.path.insert(0,'.')
import reference,torch
gather=importlib.import_module('experiments.gather.'+os.environ.get('GATHER_MODULE','gather'))
import fcntl
_gpu_lock=open('/tmp/okc-gpu.lock','w')
fcntl.flock(_gpu_lock,fcntl.LOCK_EX)
for page,heads in itertools.product((1,64),('tp8','dp')):
 data=reference.generate_input(seed=193,batch=3,ctx=257,heads=heads,layer='full',q_len=8,page=page,spread='mixed')
 expected=0
 for b in range(3):
  past=torch.arange(int(data['seq_lens'][b]),device='cuda')
  rows=data['page_table'][b,past//page].long()*page+past%page
  for name in ('k_cache','v_cache'):
   x=data[name].view(torch.int32).reshape(-1,data['config']['num_kv_heads']*16)[rows]
   expected+=x.long().sum().item()
 expected%=2**32
 for hh,pw,st in itertools.product((1,2),(0,1,2,4),(2,3,4,6)):
  out=gather.run(data,splits=7,heads=hh,stages=st,producer=pw)
  actual=out.long().sum().item()%(2**32)
  assert actual==expected,(page,heads,hh,pw,st,actual,expected)
 print('PASS native gather checksum',page,heads,32,'variants',flush=True)

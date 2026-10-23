import fcntl,sys
sys.path.insert(0,'.')
import torch,reference
from experiments.pack import pack

def validate(d,out):
 p=d['config']['page_size'];h=d['config']['num_kv_heads']
 for b,n in enumerate(d['seq_lens'].tolist()):
  rr=torch.arange(n,device='cuda');idx=d['page_table'][b,rr//p].long()*p+rr%p
  for src,dst in zip((d['k_cache'],d['v_cache']),out):
   assert torch.equal(src.view(-1,h,64).view(torch.uint8)[idx],dst[b,:n].view(torch.uint8))
   assert (dst[b,n:].view(torch.uint8)==123).all()

count=0
with open('/tmp/okc-gpu.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 for B,ctx,heads in [(1,1,'tp8'),(3,257,'tp8'),(2,2000,'dp'),(4,6000,'tp8')]:
  for page in (1,64):
   d=reference.generate_input(seed=832, batch=B,ctx=ctx,heads=heads,layer='full',page=page,q_len=8)
   out=pack.allocate(d)
   for vec in (4,8,16):
    for unroll in (1,2,4,8):
     for tensor in out:tensor.view(torch.uint8).fill_(123)
     fn=lambda:pack.run(d,out,blocks=7,unroll=unroll,vec=vec)
     fn();torch.cuda.synchronize()
     g=torch.cuda.CUDAGraph()
     with torch.cuda.graph(g):fn()
     for rep in range(3):
      # Fresh device-side byte values, page routes, and shorter (including zero) lengths.
      d['k_cache'].view(torch.uint8).random_(0,256)
      d['v_cache'].view(torch.uint8).random_(0,256)
      if rep==1:d['page_table'].copy_(d['page_table'].roll(1,0))
      if rep==2:d['seq_lens'].zero_()
      for tensor in out:tensor.view(torch.uint8).fill_(123)
      g.replay();torch.cuda.synchronize();validate(d,out);count+=1
     # Restore lengths for the next graph variant.
     d['seq_lens'].fill_(ctx)
   print('PASS',B,ctx,heads,page,flush=True)
   del d,out,g;torch.cuda.empty_cache()
print('fresh graph checks',count,flush=True)

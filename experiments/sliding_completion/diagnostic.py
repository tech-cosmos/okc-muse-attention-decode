# Offline decomposition only: modes 0/2 intentionally omit output reduction.
# They are not implementations of kernel_fn and cannot be submitted.
import fcntl,json,statistics,sys
from pathlib import Path
sys.path.insert(0,'.')
import torch,reference,submission
from torch.utils.cpp_extension import load_inline
p=Path(__file__).resolve().parent
src=(p/'prefetched.cu').read_text().replace('template<int PAGE,bool EARLY,bool FUSE>','template<int PAGE,bool EARLY,int FUSE>')
src=src.replace('  if(!last)return;','''  if(!last)return;
  if(FUSE==2) {
   if(tid==0)asm volatile("st.release.gpu.global.u32 [%0],0;"::"l"(counts+b*HKV+h):"memory");
   return;
  }''')
src=src.replace('double scale,bool fuse)','double scale,int mode)')
src=src.replace('if(page==1){if(fuse){RUN(1,true);}else{RUN(1,false);}}else{if(fuse){RUN(64,true);}else{RUN(64,false);}}','if(page==1){if(mode==2){RUN(1,2);}else if(mode){RUN(1,1);}else{RUN(1,0);}}else{if(mode==2){RUN(64,2);}else if(mode){RUN(64,1);}else{RUN(64,0);}}')
(p/'diagnostic.cu').write_text(src)
cpp='''#include <torch/extension.h>
void attention(torch::Tensor qkv,torch::Tensor cs,torch::Tensor kc,torch::Tensor vc,torch::Tensor pt,torch::Tensor seq,torch::Tensor loc,torch::Tensor ks,torch::Tensor vs,torch::Tensor pa,torch::Tensor pm,torch::Tensor counts,torch::Tensor output,int page,double scale,int mode);'''
ext=load_inline(name='muse_completion_diagnostic_v1',cpp_sources=cpp,cuda_sources=src,functions=['attention'],extra_cuda_cflags=['-gencode=arch=compute_90a,code=sm_90a','-O3','--use_fast_math','-lineinfo','-Xptxas=-v'],extra_cflags=['-O3'],verbose=True)
results=[]
with open('/tmp/okc-gpu.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 flush=torch.empty(512<<20,device='cuda',dtype=torch.uint8)
 def capture(fn):
  for _ in range(3):fn()
  torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):fn()
  return g
 def time(g):
  flush.zero_();a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True)
  a.record();g.replay();z.record();z.synchronize();return a.elapsed_time(z)*1000
 x=torch.zeros(1,device='cuda');empty=capture(lambda:x.add_(1))
 for b in(1,2):
  d=reference.generate_input(seed=107,batch=b,ctx=80000,heads='tp8',layer='sliding',page=1,dist='soft')
  h,hq,s=2,16,32;c=d['config'];pa=torch.empty((b,h,s,8,64),device='cuda');pm=torch.empty((b,h,s,8,2),device='cuda');counts=torch.zeros(b*h,device='cuda',dtype=torch.int32);out=torch.empty((b,1,hq*64),device='cuda',dtype=torch.bfloat16)
  cs=submission._cos_sin(c['rope_theta'],64,'cuda')
  def call(mode):ext.attention(d['qkv'],cs,d['k_cache'],d['v_cache'],d['page_table'],d['seq_lens'],d['cache_loc'],d['k_scale'],d['v_scale'],pa,pm,counts,out,1,c['sm_scale']*c['qk_scale_factor']*submission.LOG2E,mode)
  def check():
   assert not torch.count_nonzero(counts).item()
   w=torch.exp2(pm[...,0]-pm[...,0].amax(2,keepdim=True));a=((pa*w[...,None]).sum(2)/(w*pm[...,1]).sum(2)[...,None]*d['v_scale']).reshape(b,1,hq,64)
   gate=d['qkv'][...,(hq+2*h)*64:].float().reshape_as(a)
   output=(a*torch.rsqrt(a.square().mean(-1,keepdim=True)+1e-5)*torch.sigmoid(gate)).flatten(2).bfloat16()
   ok,msg=reference.check_implementation(d,reference.collect(d,output));assert ok,msg
  graphs=[]
  for mode in(0,2,1):
   call(mode);torch.cuda.synchronize();check();g=capture(lambda:call(mode));g.replay();torch.cuda.synchronize();check();graphs.append(g)
  for i,label in((1,'publish_only'),(2,'full_fused')):
   raw=[]
   for r in range(60):
    order=[0,i] if r%2==0 else[i,0];t={j:time(graphs[j]) for j in order};raw.append([t[0],t[i],time(empty)])
   check();e=statistics.median(v[2] for v in raw);rec={'batch':b,'variant':label,'attention_only_us':statistics.median(v[0] for v in raw)-e,'candidate_us':statistics.median(v[1] for v in raw)-e,'paired_delta_us':statistics.median(v[1]-v[0] for v in raw),'raw':raw};results.append(rec);print(json.dumps({k:v for k,v in rec.items() if k!='raw'}),flush=True)
(p/'results/diagnostic.json').write_text(json.dumps(results,indent=2)+'\n')

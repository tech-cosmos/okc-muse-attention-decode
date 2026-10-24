import sys,fcntl
from pathlib import Path
sys.path.insert(0,'.')
import torch,reference
from torch.utils.cpp_extension import load_inline
ROOT=Path(__file__).resolve().parent
s=(ROOT/'tile128.cu').read_text()
a=s.index(' float mx[2]');z=s.index('\nvoid attention',a)
s=s[:a]+'''
 if(warp==0)for(int i=0;i<64;i++)if(!(i&2)) {
  int row=lane/4,col=(i/8)*16+(lane%4)*2+(i&1)+(i&4)*2;
  pa[((b*HKV+h)*S+split)*8*128+row*128+col]=score[i];
 }
 for(int i=tid;i<8192;i+=128) {
  unsigned short vv;
  asm volatile("ld.shared.u16 %0,[%1];" : "=h"(vv) : "r"((unsigned)__cvta_generic_to_shared(V+(i ^ (((i>>6)&7)<<3)))) : "memory");
  pm[((b*HKV+h)*S+split)*8192+i]=__half2float(*(__half*)&vv);
 }
}
'''+s[z:]
(ROOT/'debug-intermediates.cu').write_text(s)
m=load_inline(name='muse_s28_intermediates',cpp_sources=(ROOT/'binding.cpp').read_text(),cuda_sources=s,functions=['attention'],extra_cuda_cflags=['-gencode=arch=compute_90a,code=sm_90a','-O3','--use_fast_math'],verbose=True)
with open('/tmp/okc-gpu.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 d=reference.generate_input(batch=1,ctx=127,heads='tp8',layer='full',page=1,q_len=1,seed=28103)
 pa=torch.empty((1,2,1,8,128),device='cuda');pm=torch.empty((1,2,1,128,64),device='cuda');qkv=d['qkv']
 m.attention(qkv,d['k_scale'],d['k_cache'],d['v_cache'],d['page_table'],d['seq_lens'],d['cache_loc'],d['k_scale'],d['v_scale'],pa,pm,1,1.,True)
 torch.cuda.synchronize()
 q=qkv[0,0,:1024].float().reshape(16,64);q=q*torch.rsqrt((q*q).mean(-1,keepdim=True)+1e-5);q=q.to(torch.float8_e4m3fn).float()
 idx=d['page_table'][0,:128].long();k=d['k_cache'][idx,0,0].float();v=d['v_cache'][idx,0,0].float();scores=q[:8]@k.T
 print('SCORE errors',[(a,float((scores[:,a:a+16]-pa[0,0,0,:,a:a+16]).abs().max())) for a in range(0,128,16)])
 print('V errors',[(a,float((v[a:a+16]-pm[0,0,0,a:a+16]).abs().max())) for a in range(0,128,16)])
 print('REFV',v[:3,:16].tolist());print('GOTV',pm[0,0,0,:3,:16].tolist())
 print('REFV64',v[64:67,:16].tolist());print('GOTV64',pm[0,0,0,64:67,:16].tolist())

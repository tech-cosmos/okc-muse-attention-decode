"""Overlap output gate sigmoid with the attention predecessor, inside kernel_fn."""
from pathlib import Path
import ast
from torch.utils.cpp_extension import load_inline
node=next(n.value for n in ast.parse(Path('submission.py').read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_NATIVE_MERGE' for t in n.targets))
base=next(ast.literal_eval(k.value) for k in node.keywords if k.arg=='cuda_sources')
cpp=next(ast.literal_eval(k.value) for k in node.keywords if k.arg=='cpp_sources')
versions={}
s=base.replace(' float vscale=*vs;',r'''
 gate.x=1.f/(1.f+expf(-gate.x));gate.y=1.f/(1.f+expf(-gate.y));
 // Require gate arithmetic before the PDL wait; gates do not depend on attention.
 asm volatile(""::"f"(gate.x),"f"(gate.y):"memory");
 float vscale=*vs;''').replace('x=x*r/(1.f+expf(-gate.x));y=y*r/(1.f+expf(-gate.y));','x=x*r*gate.x;y=y*r*gate.y;')
versions['pregate']=s
start=s.index(' float2 gate=');end=s.index(' float vscale=*vs;',start)
block=s[start:end]
block=block.replace(' float2 gate=',' gate=')
versions['pregate_warp0']=s[:start]+' float2 gate=make_float2(0.f,0.f);\n if(warp==0) {\n'+block+' }\n'+s[end:]
# The register-only arithmetic was scheduled after ACQBULK by ptxas.
# Publish to volatile shared storage, then consume the same lane's gate later.
ss=base.replace(' float vscale=*vs;',r'''
 __shared__ volatile float gates[64];
 if(warp==0) {
   gates[2*lane]=1.f/(1.f+expf(-gate.x));
   gates[2*lane+1]=1.f/(1.f+expf(-gate.y));
 }
 float vscale=*vs;''').replace('x=x*r/(1.f+expf(-gate.x));y=y*r/(1.f+expf(-gate.y));','x=x*r*gates[2*lane];y=y*r*gates[2*lane+1];')
versions['pregate_shared']=ss
mods={}
for name,source in versions.items():
 Path('experiments/reduce/session5/'+name+'.cu').write_text(source)
 mods[name]=load_inline(name='muse_reduce_s5_'+name,cpp_sources=cpp,cuda_sources=source,functions=['combine_full'],extra_cuda_cflags=['-gencode=arch=compute_90a,code=sm_90a','-O3','--use_fast_math','-lineinfo'],extra_cflags=['-O3'],verbose=False)

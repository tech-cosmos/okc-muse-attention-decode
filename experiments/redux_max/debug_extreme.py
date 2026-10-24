import ast,fcntl,json
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline
R=Path(__file__).resolve().parent

def extract(name):
 tree=ast.parse((R/(name+'.py')).read_text());node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_NATIVE_MERGE' for t in n.targets));ns={'load_inline':load_inline};exec(compile(ast.Module(body=[node],type_ignores=[]),'extract','exec'),ns);return ns['_NATIVE_MERGE']
with open('/tmp/okc-gpu.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 mods={name:extract(name) for name in ('previous','candidate')};results=[]
 q=torch.zeros((1,1,2304),device='cuda',dtype=torch.bfloat16);pa=torch.zeros((1,2,1,8,64),device='cuda');pm=torch.ones((1,2,1,8,2),device='cuda');vs=torch.ones(1,device='cuda');out=torch.empty((1,1,1024),device='cuda',dtype=torch.bfloat16)
 for maximum in (-1e30,-1e20,-1000.,-70.,0.,70.,1e20,1e30):
  pm[...,0].fill_(maximum)
  for name,mod in mods.items():
   mod.combine_full(q,pa,pm,vs,out,16,2,1,1,False);torch.cuda.synchronize()
   results.append(dict(name=name,maximum=maximum,finite=bool(torch.isfinite(out).all()),zero=bool((out==0).all())))
 (R/'debug-extreme.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))

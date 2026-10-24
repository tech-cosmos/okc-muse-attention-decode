import collections,json,re
from pathlib import Path
root=Path(__file__).resolve().parent/'results'
report={}
for stem in ('native','prefetched'):
 p=root/(stem+'.sass')
 if not p.exists():continue
 variants={}
 for name,body in re.findall(r'Function : (\S+)\n(.*?)(?=\n\s*Function :|\Z)',p.read_text(),re.S):
  ins=[]
  for line in body.splitlines():
   m=re.search(r'/\*([0-9a-f]+)\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)(.*?);',line)
   if m:ins.append((m[1],m[2],m[3].strip()))
  ops=collections.Counter(op for _,op,_ in ins)
  variants[name]={'instruction_count':len(ins),'opcodes':dict(sorted(ops.items())),'interesting':[{'pc':pc,'op':op,'args':arg} for pc,op,arg in ins if op.startswith(('LDGSTS','LDG','ATOM','MEMBAR','MUFU.RSQ','QGMMA','HGMMA','WARPGROUP','DEPBAR','BAR','STS','LDS','FFMA'))]}
 report[stem]=variants
(root/'sass-summary.json').write_text(json.dumps(report,indent=2)+'\n')
for stem,variants in report.items():
 for name,v in variants.items():
  if 'ILi1E' not in name:continue
  ops=v['opcodes'];print(stem,'fused' if 'Lb1ELb1E' in name else 'control',v['instruction_count'],{op:n for op,n in ops.items() if op.startswith(('LDGSTS','MUFU.RSQ','QGMMA','HGMMA','BAR','STS','LDS'))})
  print('first gather',next((i['pc'] for i in v['interesting'] if i['op'].startswith('LDGSTS')),None),'first rsqrt',next((i['pc'] for i in v['interesting'] if i['op']=='MUFU.RSQ'),None))

import csv,io,json,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent
report={}
def number(s):
 try:return float(s.replace(',',''))
 except (AttributeError,ValueError):return 0.
for name in ('base','qpre','plain','rope','ordered'):
 r={}
 so=ROOT/f'.build-{name}/codex_sliding_loads_{name}.so'
 resource=subprocess.check_output(['/usr/local/cuda-13.0/bin/cuobjdump','--dump-resource-usage',str(so)],text=True)
 (ROOT/f'results/resources-{name}.txt').write_text(resource)
 sass=(ROOT/f'results/{name}.sass').read_text().split('Function : _Z7slidingILi1ELb1E')[1]
 lines=[s.split(';')[0].strip() for s in sass.splitlines() if ';' in s]
 r['page1_early_instruction_sites']=len(lines)
 r['instruction_sites']={op:sum(op in s for s in lines) for op in ('QGMMA','HGMMA','LDGSTS','LDG.E.U16','STRONG.SYS','BAR.SYNC','SHFL','MUFU.EX2')}
 r['load_gather_order']=[s for s in lines[:450] if 'LDG.' in s or 'LDGSTS' in s]
 p=ROOT/f'results/profile-{name}.csv'
 if p.exists():
  s=p.read_text();rows=list(csv.DictReader(io.StringIO(s[s.index('"Address","Source"'):])))
  stalls=[k for k in rows[0] if k.startswith('stall_') and 'Not Issued' not in k]
  r['profile']={'warp_instructions_executed':sum(number(x.get('Instructions Executed')) for x in rows),'shared_excess_wavefronts':sum(number(x.get('L1 Wavefronts Shared Excessive')) for x in rows),'stall_totals':{k:sum(number(x.get(k)) for x in rows) for k in stalls},'top_stalls':[{'address':x['Address'],'source':x['Source'].strip(),'samples':number(x['# Samples']),'reason':max(stalls,key=lambda k:number(x[k]))} for x in sorted(rows,key=lambda x:number(x['# Samples']),reverse=True)[:15]]}
 for key,pattern in [('test',f'test-{name}.txt'),('fresh',f'fresh-{name}.txt')]:
  p=ROOT/'results'/pattern
  if p.exists():r[key+'_passes']=sum(x.startswith('PASS') for x in p.read_text().splitlines())
 p=ROOT/f'results/native-{name}.json'
 if p.exists():
  pairs=json.loads(p.read_text());r['comparisons']=[{k:v for k,v in a.items() if k!='raw'} for a in pairs];r['rounds']=len(pairs[0]['raw'])
 report[name]=r
(ROOT/'results/summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk not in ('load_gather_order','profile','comparisons')} for k,v in report.items()},indent=2))

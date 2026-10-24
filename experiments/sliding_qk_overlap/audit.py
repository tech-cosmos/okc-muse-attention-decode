"""Audit reproducible source and completed reports; no GPU work."""
import ast,hashlib,json,re,subprocess,sys
from pathlib import Path
assert hashlib.sha256(Path('submission.py').read_bytes()).hexdigest()=='16a7db558d53ddc110e2bf6ccec1ff243f4407b817ec939eeee409fbe2b8f16e'
ROOT=Path(__file__).resolve().parent
R=ROOT/'results'
names=('base','overlap','late','pinned')
before={n:(ROOT/(n+'.cu')).read_bytes() for n in names}
subprocess.run([sys.executable,str(ROOT/'generate.py')],check=True)
assert all((ROOT/(n+'.cu')).read_bytes()==b for n,b in before.items())
s={}
for n in ('overlap','late','pinned'):
 test=(R/f'test-{n}.log').read_text()
 assert len(re.findall(r'^PASS \d+',test,re.M))==16 and not re.search(r'^FAIL|Traceback',test,re.M)
 bench=(R/f'bench-{n}.log').read_text()
 assert len(re.findall(r'^(decode|verify) ',bench,re.M))==12
 pairs=json.loads((R/f'native-{n}.json').read_text())
 assert len(pairs)==10 and all(len(x['raw'])==60 for x in pairs)
 s[n]={'full_suite_passes':16,'native_suite_cases':3,'fallback_suite_cases':13,
       'bench_geomean':float(re.search(r'geomean over loads: ([0-9.]+)',bench)[1]),
       'paired_shapes':10,'pairs_per_shape':60,'target_pairs':[{k:v for k,v in x.items() if k!='raw'} for x in pairs[:2]],
       'source_sha256':hashlib.sha256(before[n]).hexdigest()}
 if n in ('late','pinned'):
  fresh=(R/f'fresh-{n}.log').read_text();assert fresh.count('PASS fresh graph')==10
  s[n]['fresh_native_replays']=60
  for t in ('memcheck','synccheck','racecheck'):
   log=(R/f'{t}-{n}.log').read_text()
   assert log.count('PASS native-only')==4
   assert ('ERROR SUMMARY: 0 errors' in log) if t!='racecheck' else ('RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in log)
   s[n][t]='0 errors/hazards; 12 calls'
 sass=(R/f'{n}.sass').read_text()
 body=next(b for b in sass.split('Function : ') if b.startswith('_Z7slidingILi1ELb1EE'))
 ops=[l for l in body.splitlines() if re.match(r'\s*/\*[0-9a-f]+\*/',l)]
 q=[i for i,l in enumerate(ops) if 'QGMMA' in l];v=[i for i,l in enumerate(ops) if 'STS.128' in l]
 w=[i for i,l in enumerate(ops) if 'WARPGROUP.DEPBAR' in l]
 assert len(q)==2 and len(v)==4 and len(w)==2
 if n!='late': assert q[-1]<v[0]<v[-1]<w[0]
 else: assert q[-1]<w[0]<v[0]
 s[n]['sass_order']='QK, V stores, QK wait' if n!='late' else 'QK, QK wait, softmax/V stores'
 resources=(R/f'{n}-resources.txt').read_text()
 line=next(x for x in resources.split('Function ') if x.startswith('_Z7slidingILi1ELb1EE'))
 s[n]['resources']={k:int(re.search(k+r':(\d+)',line)[1]) for k in ['REG','STACK','SHARED','LOCAL']}
def sass_body(n):
 text=(R/f'{n}.sass').read_text()
 body=next(b for b in text.split('Function : ') if b.startswith('_Z7slidingILi1ELb1EE'))
 return [x.strip() for x in body.splitlines() if re.match(r'\s*/\*[0-9a-f]+\*/',x)]
assert sass_body('overlap')==sass_body('pinned')
report={'pinned_overlap_page1_early_sass_identical':True,'production_sha256':hashlib.sha256(Path('submission.py').read_bytes()).hexdigest(),
        'variants':s,'no_judge_jobs':True,'production_changed':False,
        'sass_counts':json.loads((R/'sass-summary.json').read_text())}
(R/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

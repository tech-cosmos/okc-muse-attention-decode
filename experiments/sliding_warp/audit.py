"""Fail on incomplete evidence or accidental production/source changes."""
import ast,hashlib,json,re,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent;results=root/'results';repo=root.parents[1]
production=(repo/'submission.py').read_bytes()
assert hashlib.sha256(production).hexdigest()=='16a7db558d53ddc110e2bf6ccec1ff243f4407b817ec939eeee409fbe2b8f16e'
tree=ast.parse(production)
for stmt in tree.body:
 if isinstance(stmt,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_NATIVE_SLIDING' for t in stmt.targets):
  cu=next(k.value.value for k in stmt.value.keywords if k.arg=='cuda_sources')
  assert cu.strip()==(root/'base.cu').read_text().strip()
variants=('warp','cooperative','hybrid','warp_swizzle','cooperative_swizzle','hybrid_swizzle')
files=[f'{n}.{suffix}' for n in variants for suffix in ('cu','py')]
before={f:(root/f).read_bytes() for f in files}
subprocess.run([sys.executable,str(root/'generate.py')],check=True)
assert all((root/f).read_bytes()==before[f] for f in files)
report={'production_unchanged':True,'base_matches_embedded_native':True,'source_hashes':{f:hashlib.sha256(v).hexdigest() for f,v in before.items()},'validation':{},'bench':{},'fresh_replays':{},'sanitizers':{},'comparison_entries':{}}
for label in variants:
 log=(results/f'test-{label}.log').read_text();assert len(re.findall('^PASS ',log,re.M))==16 and 'FAIL' not in log
 report['validation'][label]={'pass':16,'native':3,'fallback':13}
 log=(results/f'bench-{label}.log').read_text();assert 'geomean over loads:' in log
 report['bench'][label]=float(log.split('geomean over loads:')[-1].strip())
 log=(results/f'fresh-{label}.log').read_text();assert len(re.findall('^PASS fresh graph ',log,re.M))==10 and 'Traceback' not in log
 report['fresh_replays'][label]=60
 for mode in (() if label.endswith('_swizzle') else ('memcheck','synccheck','racecheck')):
  log=(results/f'{mode}-{label}.log').read_text();assert len(re.findall('^PASS native-only ',log,re.M))==4 and 'Traceback' not in log
  assert ('ERROR SUMMARY: 0 errors' in log) or ('RACECHECK SUMMARY: 0 hazards' in log and '0 errors, 0 warnings' in log)
  report['sanitizers'][f'{label}-{mode}']='12 calls; zero errors/hazards'
 rows=json.loads((results/f'native-{label}.json').read_text());assert len(rows)==10 and all(len(r['raw'])==60 for r in rows)
 report['comparison_entries'][label]=len(rows)*2
report['static_sass']=json.loads((results/'sass-summary.json').read_text())
(results/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

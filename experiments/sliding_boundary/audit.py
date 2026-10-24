from pathlib import Path
import ast,hashlib,json,re,subprocess
root=Path(__file__).resolve().parent;results=root/'results';repo=root.parents[1]
report={}
assert hashlib.sha256((repo/'submission.py').read_bytes()).hexdigest()=='16a7db558d53ddc110e2bf6ccec1ff243f4407b817ec939eeee409fbe2b8f16e'
files=['base.cu','pruned.cu','half.cu','binding.txt'];before={f:(root/f).read_bytes() for f in files}
subprocess.run(['python',str(root/'generate.py')],check=True)
assert all((root/f).read_bytes()==before[f] for f in files)
report['source_hashes']={f:hashlib.sha256(v).hexdigest() for f,v in before.items()}
report['production_unchanged']=True
report['validation']={}
for label in ('pruned','half','boundary'):
 log=(results/f'validation-{label}.txt').read_text();assert len(re.findall('^PASS ',log,re.M))==16 and 'FAIL' not in log
 report['validation'][label]=16
 bench=(results/f'bench-{label}.txt').read_text();assert 'geomean over loads:' in bench
 report.setdefault('bench',{})[label]=float(bench.split('geomean over loads:')[-1].strip())
report['fresh_replays']={}
for label in ('base','pruned','half'):
 log=(results/f'fresh-{label}.txt').read_text();assert len(re.findall('^PASS fresh graph ',log,re.M))==10 and 'Traceback' not in log
 report['fresh_replays'][label]=60
report['sanitizers']={}
for label in ('pruned','half'):
 for mode in ('memcheck','synccheck','racecheck'):
  log=(results/f'{mode}-{label}.txt').read_text();assert len(re.findall('^PASS native-only ',log,re.M))==4 and 'Traceback' not in log
  assert ('ERROR SUMMARY: 0 errors' in log) or ('RACECHECK SUMMARY: 0 hazards' in log and '0 errors, 0 warnings' in log)
  report['sanitizers'][f'{label}-{mode}']='12 calls; zero errors/hazards'
report['comparison_entries']={}
for name in ('initial-base','nearby-base','confirm-base','edge-base','initial-pruned','structural-pruned','structural-half'):
 rows=json.loads((results/f'{name}.json').read_text());assert all(len(r['raw'])==60 for r in rows)
 report['comparison_entries'][name]=len(rows)
report['static_sass']=json.loads((results/'sass-summary.json').read_text())
subprocess.run(['git','apply','--check',str(root/'proposed-dispatch.patch')],check=True,cwd=repo)
report['pending_patch_applies']=True
(results/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

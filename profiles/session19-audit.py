"""Audit session19 experiment provenance, completed validation and final source."""
import ast,hashlib,json,subprocess,tempfile
from pathlib import Path
BRANCH='codex-exp/cuda-sliding-loads'
PREFIX='experiments/sliding_loads/'
def read(name):
 return subprocess.check_output(['git','show',BRANCH+':'+PREFIX+name],text=True)
def canon(s):return '\n'.join(x.rstrip() for x in s.splitlines()).rstrip()+'\n'
source=Path('submission.py').read_text()
mod=ast.parse(source)
call=next(n.value for n in mod.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_NATIVE_SLIDING' for t in n.targets))
assert canon(ast.literal_eval(next(k.value for k in call.keywords if k.arg=='cuda_sources')))==read('base.cu')
with tempfile.TemporaryDirectory(prefix='.session19-audit-',dir='.') as temp:
 p=Path(temp)
 for n in ('base.cu','base.py'):(p/n).write_text(read(n))
 exec(compile(read('generate.py'),'archived_generate.py','exec'),{'__file__':str(p/'generate.py')})
 for variant in ('qpre','plain','rope','ordered'):
  for ext in ('.cu','.py'):assert canon((p/(variant+ext)).read_text())==read(variant+ext)
report={'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'experiment_branch':BRANCH,'variants':{}}
for name in ('qpre','plain','rope','ordered'):
 test=read(f'results/test-{name}.txt');fresh=read(f'results/fresh-{name}.txt');pairs=json.loads(read(f'results/native-{name}.json'))
 assert sum(s.startswith('PASS ') for s in test.splitlines())==16 and 'FAIL' not in test
 assert sum(s.startswith('PASS fresh') for s in fresh.splitlines())==10
 assert len(pairs)==10 and all(len(x['raw'])==60 for x in pairs)
 assert read(f'results/resources-{name}.txt').count('REG:74 STACK:0 SHARED:17408 LOCAL:0')==4
 report['variants'][name]={'suite_passed':16,'fresh_replays':60,'paired_shapes':10,'rounds':60}
 if name in ('rope','ordered'):
  for tool in ('memcheck','synccheck','racecheck'):
   log=read(f'results/{tool}-{name}.txt');assert log.count('PASS native-only')==4
   assert ('0 hazards displayed (0 errors, 0 warnings)' if tool=='racecheck' else 'ERROR SUMMARY: 0 errors') in log
  report['variants'][name]['sanitizers']='memcheck/synccheck/racecheck: zero errors/hazards'
confirm=json.loads(read('results/native-rope-confirm.json'));assert len(confirm)==10 and all(len(x['raw'])==120 for x in confirm)
report['rope_confirmation_B1_delta_us']=confirm[0]['paired_delta_us']
for filename,count,prefix in [('session19-integrated-test.txt',16,'PASS '),('session19-integrated-fresh.txt',8,'PASS fresh')]:
 s=Path('profiles',filename).read_text();assert sum(x.startswith(prefix) for x in s.splitlines())==count and 'FAIL' not in s
pairs=json.loads(Path('profiles/session19-integrated-pairs.json').read_text());assert len(pairs)==5 and all(len(x['raw'])==60 for x in pairs)
assert 'batch * hkv <= 16' in source and 'def _verify_gluon' in source
score=json.loads(Path('profiles/session19-score-mid4.json').read_text())
assert score['status']=='done' and score['result']['passed'] and not score['result']['error']
assert not score['result']['checks']['flagged'] and not score['result']['checks']['rejected']
report.update(integrated_tests=16,integrated_fresh_replays=24,integrated_pair_shapes=5,scored_job=score['id'],speedup=score['result']['speedup'],new_check=json.loads(Path('profiles/session19-new-check.json').read_text())['job_id'])
Path('profiles/session19-validation-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

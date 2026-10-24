"""Audit the late main824e52a integration, separately from initial check3e786 source."""
import ast
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
final=(ROOT/'final_snapshot.py').read_bytes()
initial=(ROOT/'initial_snapshot.py').read_bytes()
main=(ROOT/'final_main_snapshot.py').read_bytes()
assert hashlib.sha256(final).hexdigest()=='1bcf42aa6edf7f25a02117e3eabebb6f5b2e69967359137e015b2f79d445a548'
assert hashlib.sha256(initial).hexdigest()=='6f0fcb357146bbfd4e69e3f9d8a3ea2d9968f09c26b5876a0aa83af63ebe8c05'
assert Path('submission.py').read_bytes()==final
parsed={k:ast.parse(v) for k,v in dict(final=final,initial=initial,main=main).items()}
def functions(tree):
 return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)}
f={k:functions(v) for k,v in parsed.items()}
shared=f['final'].keys()&f['main'].keys()-{'kernel_fn'}
assert all(f['final'][n]==f['main'][n] for n in shared)
assert f['final']['kernel_fn']==f['initial']['kernel_fn']
changed=sorted(n for n in f['final'] if f['final'][n]!=f['initial'].get(n))
assert changed==['_gw_consumer_cv','_gw_verify','_plan','_verify_gluon'],changed
for name in ('_NATIVE_MERGE','_NATIVE_SLIDING'):
 def get(tree):
  return next(ast.dump(x,include_attributes=False) for x in tree.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in x.targets))
 assert get(parsed['final'])==get(parsed['initial'])
rows=json.loads((ROOT/'results'/'final-integration.json').read_text());assert len(rows)==7
assert all(len(r['raw'])==80 for r in rows)
assert all(r['dispatch']=='gluon_fallback' for r in rows)
fresh=json.loads((ROOT/'results'/'fresh-final-integration.json').read_text());assert len(fresh)==7
assert all(r['replays_per_implementation']==4 for r in fresh)
log=(ROOT/'results'/'fresh-final-integration.log').read_text()
assert log.count('PASS fresh ')==7 and log.count('replay ')==56 and 'Traceback' not in log
log=(ROOT/'results'/'final-test.log').read_text()
assert sum(l.startswith('PASS ') for l in log.splitlines())==16 and 'FAIL' not in log and 'Traceback' not in log
result=dict(final_sha256=hashlib.sha256(final).hexdigest(),initial_sha256=hashlib.sha256(initial).hexdigest(),
 main_sha256=hashlib.sha256(main).hexdigest(),all_shared_helpers_including_nested=len(shared),
 changed_helpers=changed,native_source_and_dispatch_unchanged=True,standard_tests=16,
 compared_shapes=7,pairs=560,reference_checks=42,fresh_replays=56,
 judge_qualification='NOT covered by check3e786: that request has the earlier initial snapshot')
(ROOT/'results'/'final-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

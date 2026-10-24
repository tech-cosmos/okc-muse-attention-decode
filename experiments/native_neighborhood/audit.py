"""CPU-only evidence/source audit. Run from repository root after comparisons."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
sources={n:(ROOT/f'{n}_snapshot.py').read_bytes() for n in ('main','previous')}
sources['current']=Path('submission.py').read_bytes()
expected={'main':'01a58b8042e2c8fbeba1b351f90ba487973d956fdd8e3bfe4d5c2a773a98be9f',
          'previous':'6b171beb8ed5e2a306ff67c3820ba7e83943f73c1a73d7b06eeeb630e0d0eecf',
          'current':'6f0fcb357146bbfd4e69e3f9d8a3ea2d9968f09c26b5876a0aa83af63ebe8c05'}
for n,s in sources.items():assert hashlib.sha256(s).hexdigest()==expected[n],n
nodes={n:ast.parse(s) for n,s in sources.items()}
functions={n:{x.name:ast.dump(x,include_attributes=False) for x in tree.body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef))} for n,tree in nodes.items()}
shared=functions['main'].keys()&functions['current'].keys()-{'kernel_fn'}
assert all(functions['main'][n]==functions['current'][n] for n in shared)
changed=[n for n in functions['previous'] if functions['previous'][n]!=functions['current'].get(n)]
assert changed==['_plan'],changed
# Native extension definitions and the main dispatcher retain exact ASTs.
for name in ('_NATIVE_MERGE','_NATIVE_SLIDING'):
 def find(tree):
  return next(ast.dump(x,include_attributes=False) for x in tree.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in x.targets))
 assert find(nodes['previous'])==find(nodes['current']),name
counts={}; rows=[]
for name,num,rounds in [('integration',13,80),('neighborhood',44,80),('integration-confirmation',3,120),('neighborhood-confirmation',9,120)]:
 entries=json.loads((ROOT/'results'/f'{name}.json').read_text());assert len(entries)==num
 assert len({r['index'] for r in entries})==num
 assert all(r['shape']['layer']=='full' and r['shape']['q_len']==8 for r in entries)
 for row in entries:
  assert len(row['raw'])==rounds
  assert all(len(t)==3 and all(v>0 for v in t) for t in row['raw'])
 counts[name]=dict(shapes=num,pairs=num*rounds,reference_checks=num*6)
 rows+=entries
 assert 'Traceback' not in (ROOT/'results'/f'{name}.log').read_text()
for name in ('integration',):
 entries=json.loads((ROOT/'results'/f'fresh-{name}.json').read_text());assert len(entries)==13
 assert all(e['replays_per_implementation']==4 for e in entries)
 counts['fresh_'+name]=dict(shapes=13,replays=104)
log=(ROOT/'results'/'fresh-integration.log').read_text()
assert log.count('PASS fresh ')==13 and log.count('replay ')==104 and 'Traceback' not in log
log=(ROOT/'results'/'integrated-test.log').read_text()
assert sum(line.startswith('PASS ') for line in log.splitlines())==16 and 'FAIL' not in log and 'Traceback' not in log
counts['integrated_standard_suite']=dict(passed=16)
summary=dict(source_hashes=expected,shared_helpers=len(shared),changed_helpers=changed,counts=counts,
             native_source_and_dispatch_unchanged=True)
(ROOT/'results'/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))

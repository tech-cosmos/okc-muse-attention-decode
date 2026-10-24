"""CPU-only provenance, generated-source, exact-bit and graph-result audit."""
import ast,hashlib,json,re,statistics,subprocess
from pathlib import Path
from generate import generate
from bitprobe import data
r=Path(__file__).resolve().parent
old=(r/'previous.py').read_text();new=(r/'candidate.py').read_text()
assert old==subprocess.check_output(['git','show','023f00c:submission.py'],text=True)
assert new==generate()
def strip(tree):
 return ast.dump(ast.Module(body=[n for n in tree.body if not(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_NATIVE_MERGE' for t in n.targets))],type_ignores=[]))
assert strip(ast.parse(old))==strip(ast.parse(new))
# Main-owned helper identities, including nested definitions.
main=ast.parse(subprocess.check_output(['git','show','06b7755:submission.py'],text=True))
helpers={n.name:ast.dump(n) for n in ast.walk(main) if isinstance(n,ast.FunctionDef) and n.name!='kernel_fn'}
ours={n.name:ast.dump(n) for n in ast.walk(ast.parse(new)) if isinstance(n,ast.FunctionDef)}
assert all(ours[name]==body for name,body in helpers.items())
res=dict(production_sha256=hashlib.sha256(old.encode()).hexdigest(),candidate_sha256=hashlib.sha256(new.encode()).hexdigest(),only_native_merge_changed=True,main_helpers_identical=len(helpers))
logs=(r/'suite.log').read_text();assert sum(x.startswith('PASS ') for x in logs.splitlines())==16
assert 'FAIL ' not in logs and 'geomean over loads:' in logs
res['suite_cases']=16
pairs=checks=0
for f in sorted(r.glob('pairs-*.json')):
 rows=json.loads(f.read_text());assert len(rows)==(6 if f.name=='pairs-confirm.json' else 16)
 for row in rows:
  raw=row['raw'];assert len(raw)==80
  cost=statistics.median(x[2] for x in raw)
  assert row['previous_us']==statistics.median(x[0] for x in raw)-cost
  assert row['merged_us']==statistics.median(x[1] for x in raw)-cost
  assert row['paired_delta_us']==statistics.median(x[1]-x[0] for x in raw)
  pairs+=len(raw);checks+=6
res.update(adjacent_pairs=pairs,comparison_references=checks)
fresh=json.loads((r/'fresh.json').read_text());assert len(fresh)==16
res['fresh_wrapper_replays']=sum(row['replays_per_source']*2 for row in fresh)
compare=json.loads((r/'pairs-compare.json').read_text())
res['fresh_changed_native_replays']=4*sum(row['native'] for row in compare)
for label in ('plain','synccheck'):
 syn=json.loads((r/f'synthetic-{label}.json').read_text());assert len(syn)==12
 assert all(row['replays']==6 for row in syn)
 res[f'synthetic_{label}_replays']=sum(row['replays'] for row in syn)
 res[f'synthetic_{label}_max_rel_l2']=max(max(row['relative_l2']) for row in syn)
assert 'ERROR SUMMARY: 0 errors' in (r/'synccheck.log').read_text()
values,expected=data();probe=json.loads((r/'bitprobe.json').read_text())
assert values.size==probe['finite_values'] and len(values)==probe['rows']
res['bitprobe']=probe
resources={}
for label in ('previous','candidate'):
 sass=(r/(label+'.sass')).read_text();functions=sass.split('Function : ')[1:];assert len(functions)==2
 resources[label]=[]
 for fn in functions:
  lines=[re.sub(r'/\*.*?\*/','',line).strip() for line in fn.splitlines() if ';' in line and '/*' in line]
  if 'ELb1E' in fn.splitlines()[0]:
   wait=next(i for i,line in enumerate(lines) if 'ACQBULK' in line)
   loads=[i for i,line in enumerate(lines) if 'LDG.E.64.STRONG.GPU' in line]
   assert len(loads)==17 and min(loads)>wait
  redux=sum('REDUX.MAX.S32' in x for x in lines)
  assert redux==(label=='candidate')
  resources[label].append(dict(name=fn.splitlines()[0],static_instructions=len(lines),redux_sites=redux,shuffle_sites=sum('SHFL.' in x for x in lines),float_max_sites=sum('FMNMX' in x for x in lines)))
 txt=(r/(label+'-resources.txt')).read_text();assert txt.count('REG:48 STACK:0 SHARED:3200 LOCAL:0')==2
res['sass']=resources
for old_fn,new_fn in zip(resources['previous'],resources['candidate']):
 assert new_fn['shuffle_sites']==old_fn['shuffle_sites']-5
 assert new_fn['float_max_sites']==old_fn['float_max_sites']-5
res['source_and_results_pass']=True
(r/'audit.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2))

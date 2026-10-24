"""CPU-only provenance, numerical-results and paired timing audit."""
import ast,hashlib,json,statistics,subprocess,re
from pathlib import Path
r=Path(__file__).resolve().parent
old=(r/'previous.py').read_text();new=(r/'candidate.py').read_text()
assert old==subprocess.check_output(['git','show','ce81a25:submission.py'],text=True)
subprocess.run(['python',str(r/'generate.py')],check=True)
assert new==(r/'candidate.py').read_text()
a,b=ast.parse(old),ast.parse(new)
def stripped(tree):
    return ast.dump(ast.Module(body=[n for n in tree.body if not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_NATIVE_MERGE' for t in n.targets))],type_ignores=[]))
assert stripped(a)==stripped(b)
res=dict(production_sha256=hashlib.sha256(old.encode()).hexdigest(),candidate_sha256=hashlib.sha256(new.encode()).hexdigest(),only_native_merge_changed=True)
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
syn=json.loads((r/'synthetic-plain.json').read_text());assert len(syn)==12
res['synthetic_replays']=sum(row['replays'] for row in syn)
res['synthetic_max_rel_l2']=max(max(row['relative_l2']) for row in syn)
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
        resources[label].append(dict(name=fn.splitlines()[0],static_instructions=len(lines),reciprocal_sites=sum('MUFU.RCP' in x for x in lines),exponential_sites=sum('MUFU.EX2' in x for x in lines)))
    txt=(r/(label+'-resources.txt')).read_text()
    assert txt.count('REG:48 STACK:0 SHARED:3200 LOCAL:0')==2
res['sass']=resources
res['source_and_results_pass']=True
(r/'audit.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2))

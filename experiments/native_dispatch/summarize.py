"""CPU-only evidence audit; no kernel changes or GPU work."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
production=Path('submission.py').read_text()
main=(ROOT/'main_snapshot.py').read_text()
sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
assert sha(production)=='6b171beb8ed5e2a306ff67c3820ba7e83943f73c1a73d7b06eeeb630e0d0eecf'

def functions(s):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
p,m=functions(production),functions(main)
common=set(p)&set(m)-{'kernel_fn'}
assert all(p[n]==m[n] for n in common)
assert set(p)==set(m)
all_rows=[]
files={}
for name,count,rounds in [('initial',50,80),('split-boundary',12,80),('verify-split-boundary',2,80),('confirmation',8,120)]:
    rows=json.loads((ROOT/'results'/f'{name}.json').read_text())
    assert len(rows)==count,(name,len(rows))
    assert all(len(r['raw'])==rounds for r in rows)
    assert all(len(r['raw'])%2==0 for r in rows)
    files[name]=dict(shapes=len(rows),rounds_per_shape=rounds,
                     dispatch_counts=dict(Counter(r['dispatch'] for r in rows)))
    if name!='confirmation':all_rows.extend(rows)
assert sorted(r['index'] for r in all_rows)==list(range(64))
fresh=json.loads((ROOT/'results'/'fresh.json').read_text())
assert len(fresh)==24
assert all(r['replays_per_implementation']==4 for r in fresh)
summary=dict(production_sha256=sha(production),main_sha256=sha(main),main_commit='3daf4a4',
             common_function_asts_identical=sorted(common),timing_files=files,
             total_adjacent_pairs=sum(v['shapes']*v['rounds_per_shape'] for v in files.values()),
             reference_checks_in_timing=3*2*sum(v['shapes'] for v in files.values()),
             fresh_shapes=len(fresh),fresh_replays_per_implementation=4*len(fresh),
             fresh_codex_dispatch_counts=dict(Counter({k:4*v for k,v in Counter(r['dispatch'] for r in fresh).items()})),
             initial_dispatch_ranges={})
for dispatch in sorted(set(r['dispatch'] for r in all_rows)):
    rows=[r for r in all_rows if r['dispatch']==dispatch]
    summary['initial_dispatch_ranges'][dispatch]=dict(count=len(rows),
       median_delta_min_us=min(r['paired_delta_us'] for r in rows),
       median_delta_max_us=max(r['paired_delta_us'] for r in rows))
(ROOT/'results'/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))

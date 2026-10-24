"""CPU-only integration provenance and measurement accounting."""
import ast,hashlib,json,statistics,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent
sha=lambda x:hashlib.sha256(x).hexdigest()
p,n,u=[(R/f).read_bytes() for f in ('previous.py','merged.py','upstream.py')]
assert n==Path('submission.py').read_bytes()
assert p==subprocess.check_output(['git','show','18aa622:submission.py'])
assert u==subprocess.check_output(['git','show','eef9721:submission.py'])
def nodes(src):
    return {x.name:ast.dump(x,include_attributes=False) for x in ast.walk(ast.parse(src)) if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
pa,na,ua=map(nodes,(p,n,u))
shared=sorted(set(na)&set(ua)-{'kernel_fn'})
assert all(na[k]==ua[k] for k in shared)
assert pa['kernel_fn']==na['kernel_fn']
# Native compile payloads and every extension option are bit-for-bit AST-identical.
def extension(src,name):
    return next(ast.dump(x,include_attributes=False) for x in ast.parse(src).body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in x.targets))
for name in ('_NATIVE_MERGE','_NATIVE_SLIDING'):assert extension(p,name)==extension(n,name)
result=dict(previous_sha256=sha(p),merged_sha256=sha(n),upstream_sha256=sha(u),upstream_commit='eef9721',shared_AST_nodes=shared,native_extensions_and_dispatch_unchanged=True,runs={})
count=checks=0
for mode,expected in [('compare',15),('confirm',12),('scored',12)]:
    rows=json.loads((R/('pairs-'+mode+'.json')).read_text());assert len(rows)==expected
    assert (R/(mode+'.log')).read_text().count('PASS ')==6*expected
    for row in rows:
        raw=row['raw'];assert len(raw)==80 and all(len(x)==3 and all(v>0 for v in x) for x in raw)
        base=statistics.median(x[2] for x in raw)
        assert abs(row['previous_us']-(statistics.median(x[0] for x in raw)-base))<1e-10
        assert abs(row['merged_us']-(statistics.median(x[1] for x in raw)-base))<1e-10
        assert abs(row['paired_delta_us']-statistics.median(x[1]-x[0] for x in raw))<1e-10
    count+=80*expected;checks+=6*expected
    result['runs'][mode]=dict(cases=expected,pairs=80*expected,reference_checks=6*expected,gfuse_cases=sum(r['gfuse'] for r in rows),control_deltas_us=[r['paired_delta_us'] for r in rows if not r['gfuse']])
fresh=json.loads((R/'fresh.json').read_text());assert len(fresh)==15 and all(r['replays_per_source']==4 for r in fresh)
assert (R/'fresh.log').read_text().count('PASS ')==120
counters=json.loads((R/'counters.json').read_text());assert len(counters)==36 and all(r['output_cache_pass'] for r in counters)
assert (R/'counters.log').read_text().count('PASS ')==36
assert (R/'suite.log').read_text().count('PASS ')==16
assert 'FAIL' not in (R/'suite.log').read_text()
result.update(total_pairs=count,total_timing_reference_checks=checks,fresh_replays=120,counter_boundary_replays=36,standard_suite=16)
(R/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

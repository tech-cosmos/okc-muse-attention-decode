"""CPU-only provenance, shared-helper and paired-result audit."""
import ast,hashlib,json,statistics,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent
sha=lambda x:hashlib.sha256(x).hexdigest()
native=(R/'native.py').read_bytes();upstream=(R/'upstream.py').read_bytes()
assert native==Path('submission.py').read_bytes()
assert upstream==subprocess.check_output(['git','show','463b0d5:submission.py'])
def functions(src):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(src).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
u,n=functions(upstream),functions(native)
shared=sorted(set(u)&set(n)-{'kernel_fn'})
assert all(u[k]==n[k] for k in shared)
result={'native_sha256':sha(native),'upstream_sha256':sha(upstream),'upstream_commit':'463b0d5','shared_AST_nodes':shared,'runs':{}}
count=0
for mode,expected in [('compare',27),('confirm',15),('scored',12)]:
    rows=json.loads((R/('pairs-'+mode+'.json')).read_text());assert len(rows)==expected
    log=(R/(mode+'.log')).read_text();assert log.count('PASS ')==6*expected
    for row in rows:
        raw=row['raw'];assert len(raw)==80 and all(len(x)==3 and all(v>0 for v in x) for x in raw)
        base=statistics.median(x[2] for x in raw)
        assert abs(row['previous_us']-(statistics.median(x[0] for x in raw)-base))<1e-10
        assert abs(row['merged_us']-(statistics.median(x[1] for x in raw)-base))<1e-10
        assert abs(row['paired_delta_us']-statistics.median(x[1]-x[0] for x in raw))<1e-10
    count+=80*expected
    result['runs'][mode]={'cases':expected,'pairs':80*expected,'reference_checks':6*expected,
         'full_native_cases':sum(r['native_full'] for r in rows),
         'sliding_native_cases':sum(r['native_sliding'] for r in rows),
         'control_deltas_us':[r['paired_delta_us'] for r in rows if not r['native_full'] and not r['native_sliding']]}
fresh=json.loads((R/'fresh.json').read_text());assert len(fresh)==27 and all(r['replays_per_source']==4 for r in fresh)
assert (R/'fresh.log').read_text().count('PASS ')==216
result.update(total_pairs=count,total_timing_reference_checks=6*54,fresh_replays=216)
# Net observed weighted benefit is descriptive; unchanged-control noise remains visible.
rows=json.loads((R/'pairs-scored.json').read_text())
loads=[]
for i in range(0,12,2):
    a,b=rows[i:i+2];old=46*a['previous_us']+16*b['previous_us'];new=46*a['merged_us']+16*b['merged_us']
    loads.append({'batch':a['shape']['batch'],'q_len':a['shape']['q_len'],'upstream_weighted_us':old,'native_weighted_us':new,'ratio':old/new})
result['scored_local_loads']=loads
(R/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

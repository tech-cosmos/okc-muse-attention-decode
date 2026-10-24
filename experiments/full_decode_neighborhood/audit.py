"""CPU-only audit of frozen sources, complete graph samples and validation counts."""
import ast
import hashlib
import json
import re
import statistics
import subprocess
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((RESULTS/'source-manifest.json').read_text())
for name,h in manifest['sources'].items():assert sha(ROOT/name)==h,name
for name in ('candidate.py','binding.cpp','tile128.cu','production_snapshot.py'):
    old=subprocess.check_output(['git','show','081f126:experiments/full_decode_tile/'+name])
    assert hashlib.sha256(old).hexdigest()==sha(ROOT/name),name
assert sha(Path('submission.py'))==sha(ROOT/'production_snapshot.py')

# Only the extra native-merge conditional and default width distinguish this
# composition from the frozen wrapper; attention and the fallback launch match.
old=ast.parse((ROOT/'candidate.py').read_text())
new=ast.parse((ROOT/'native_merge_control.py').read_text())
of=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name=='kernel')
nf=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name=='kernel')
assert len(nf.body)==len(of.body)+1
extra=[i for i,n in enumerate(nf.body) if isinstance(n,ast.If) and '_NATIVE_MERGE' in ast.unparse(n)]
assert len(extra)==1
del nf.body[extra[0]]
nf.args.defaults=of.args.defaults
assert ast.dump(nf,include_attributes=False)==ast.dump(of,include_attributes=False)

summary=[];pairs=0;checks=0
for name,count,rounds,per_shape in [('initial',74,60,6),('confirmation',22,100,6),('merge-control',32,80,9),('merge-confirmation',32,80,9)]:
    rows=json.loads((RESULTS/(name+'.json')).read_text());assert len(rows)==count
    indices={r['index'] for r in rows}
    assert len(indices)==(16 if name.startswith('merge-') else count)
    if name=='initial':assert indices==set(range(74))
    log=(RESULTS/(name+'.log')).read_text()
    expected=len(indices)*per_shape
    assert log.count('CHECK ')==expected,(name,log.count('CHECK '))
    assert ' False ' not in log and 'Traceback' not in log
    checks+=expected
    for r in rows:
        assert len(r['raw'])==rounds
        raw=np.asarray(r['raw']);assert np.isfinite(raw).all()
        delta=raw[:,1]-raw[:,0]
        assert abs(float(np.median(delta))-r['paired_delta_us'])<1e-10
        # Bootstrap neighboring AB/BA pairs as blocks. Within-run uncertainty only.
        blocks=delta.reshape(-1,2)
        rng=np.random.default_rng(29601)
        boot=np.median(blocks[rng.integers(len(blocks),size=(2000,len(blocks)))].reshape(2000,-1),axis=1)
        small={k:v for k,v in r.items() if k!='raw'}
        small.update(run=name,paired_median_95pct_us=np.percentile(boot,[2.5,97.5]).tolist())
        summary.append(small);pairs+=rounds
fresh=(RESULTS/'fresh.log').read_text()
assert fresh.count("CHECK ('fresh'")==64 and ' False ' not in fresh and 'Traceback' not in fresh
fr=json.loads((RESULTS/'fresh.json').read_text());assert len(fr)==16
composed=(RESULTS/'merge-fresh.log').read_text()
assert composed.count("CHECK ('composed-fresh'")==64
assert 'FRESH_TOTAL 64' in composed and ' False ' not in composed and 'Traceback' not in composed
suite=(RESULTS/'merge-suite.log').read_text()
assert len(re.findall(r'^PASS \d+ ',suite,re.M))==16 and 'FAIL' not in suite
assert suite.count('H100 time vs SGLang-on-H200')==1
merge=json.loads((RESULTS/'merge-control.json').read_text())
active=sum(r['uses_native_merge'] for r in merge if r['baseline']=='parent')
result=dict(production_sha256=sha(Path('submission.py')),inherited_experiment='081f126',
            frozen_sources=manifest['sources'],timing_shapes=74,confirmation_entries=22,
            merge_control_shapes=16,merge_control_runs=2,merge_control_active_native_shapes=active,
            adjacent_pairs=pairs,comparison_reference_checks=checks,
            fresh_original_tile_replays=64,fresh_composed_replays=64,
            fresh_composed_active_native_merge_replays=active*4,
            composed_suite_passes=16,composed_full_benches=1,
            production_changed=False,judge_requests=0,new_sanitizer_runs=0)
(RESULTS/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(RESULTS/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

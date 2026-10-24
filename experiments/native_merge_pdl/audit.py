"""CPU-only reproducibility and result-count audit."""
import ast
import hashlib
import json
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def nodes(p):
    tree=ast.parse(p.read_text())
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(tree)
            if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}

prod=ROOT/'production_snapshot.py';main=ROOT/'main_snapshot.py'
assert prod.read_bytes()==subprocess.check_output(['git','show','d8bab0a:submission.py'])
assert main.read_bytes()==subprocess.check_output(['git','show','824e52a:submission.py'])
pn,mn=nodes(prod),nodes(main)
shared=sorted(set(pn)&set(mn)-{'kernel_fn'})
assert all(pn[k]==mn[k] for k in shared)
summary={'production_sha256':digest(prod),'main_sha256':digest(main),'shared_helpers':shared,
         'production_native_source_changed':False,'triton_source_or_plan_changed':False}
count=0;refs=0
for name,n,rounds in [('initial',20,80),('confirmation',8,120)]:
    entries=json.loads((ROOT/'results'/f'{name}.json').read_text());assert len(entries)==n
    for row in entries:
        assert 64<row['splits']<=128
        assert len(row['contrasts'])==4
        for contrast in row['contrasts']:
            assert len(contrast['raw'])==rounds
            assert all(len(r)==3 and all(t>0 for t in r) for r in contrast['raw'])
            count+=rounds
    refs+=n*4*3
    summary[name]={'shapes':n,'pairs':n*4*rounds,'reference_checks':n*4*3}
entries=json.loads((ROOT/'results'/'fresh.json').read_text());assert len(entries)==20
assert all(row['replays_per_implementation']==4 for row in entries)
assert (ROOT/'results'/'fresh.log').read_text().count('PASS replay ')==320
summary.update(total_adjacent_pairs=count,timing_reference_checks=refs,fresh_replays=320)
timeline=json.loads((ROOT/'results'/'timeline-intervals.json').read_text())
assert len(timeline)==24
for r in timeline:
    assert len(r['kernels'])==2
    assert (r['kernel_overlap_us']==0) == (':serial_' in r['label'])
assert 'PASS 16 output/cache checks' in (ROOT/'results'/'timeline.log').read_text()
occ=json.loads((ROOT/'results'/'occupancy.json').read_text());assert len(occ)==8
for case,regs,ctas in [(3,170,2),(13,168,3)]:
    selected=[r for r in occ if r['case']==case];assert len(selected)==4
    assert len({r['sass_text_sha256'] for r in selected})==1
    for r in selected:
        assert r['registers']==regs and r['cuda_max_active_ctas']==ctas
        assert r['shared_bytes']==62464 and r['sm_count']==132
        assert digest(ROOT/'results'/f"case{case}-{r['mode']}.sass")==r['sass_text_sha256']
summary.update(instrumented_graph_replays=24,timeline_reference_checks=16,
               final_occupancy_reference_checks=8,attention_sass_identical_across_modes=True,
               residency_page1=2,residency_page64=3)
print(json.dumps(summary,indent=2))
(ROOT/'results'/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')

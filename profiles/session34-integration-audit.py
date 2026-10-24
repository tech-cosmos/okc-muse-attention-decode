"""CPU-only source and validation audit for main19b8ade integration."""
import ast,hashlib,json,re,statistics,subprocess
from pathlib import Path
R=Path('profiles')
current=Path('submission.py').read_text();previous=(R/'session34-previous.py').read_text()
main=subprocess.check_output(['git','show','44e4e25:submission.py'],text=True)
sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
assert sha(current)=='62fefe79bd913e6dc1b12388ad7bec50139096cb07c7b61918b8031c0c93a86b'
assert sha(previous)=='1bcf42aa6edf7f25a02117e3eabebb6f5b2e69967359137e015b2f79d445a548'
def funcs(s):return {n.name:ast.dump(n) for n in ast.walk(ast.parse(s)) if isinstance(n,ast.FunctionDef)}
c,p,m=map(funcs,(current,previous,main))
shared=[k for k in m if k!='kernel_fn']
assert len(shared)==20 and all(c[k]==m[k] for k in shared)
assert c['kernel_fn']==p['kernel_fn']
for name in ('_NATIVE_MERGE','_NATIVE_SLIDING'):
    def assignment(s):
        return ast.dump(next(n for n in ast.parse(s).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)))
    assert assignment(current)==assignment(previous)
suite=(R/'session34-integrated-suite.log').read_text()
assert len(re.findall(r'^PASS \d+ ',suite,re.M))==16
assert len(re.findall(r'^(decode|verify) ',suite,re.M))==12
pairs=json.loads((R/'session34-integrated-pairs.json').read_text());fresh=json.loads((R/'session34-integrated-fresh.json').read_text())
assert len(pairs)==len(fresh)==10
for row in pairs:
    assert len(row['raw'])==80
    assert abs(row['paired_delta_us']-statistics.median(b-a for a,b,e in row['raw']))<1e-12
assert all(row['replays_per_source']==4 for row in fresh)
report=dict(previous_sha256=sha(previous),current_sha256=sha(current),main_commit='44e4e25',native_unchanged=True,
            matching_main_helpers=20,unchanged_dispatch=True,suite_passes=16,bench_shapes=12,
            paired_comparisons=800,paired_reference_checks=60,fresh_replays=80,
            results=[{k:v for k,v in row.items() if k!='raw'} for row in pairs])
(R/'session34-integrated-audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

"""CPU audit of frozen source changes, measurements and validation counts."""
from pathlib import Path
import ast,hashlib,json,statistics,re
ROOT=Path(__file__).resolve().parent
s=(ROOT/'production_snapshot.py').read_text();v=(ROOT/'neighbor.py').read_text()
assert hashlib.sha256(s.encode()).hexdigest()=='1bcf42aa6edf7f25a02117e3eabebb6f5b2e69967359137e015b2f79d445a548'
assert s==Path('submission.py').read_text()
expected=s.replace('const int b=blockIdx.y,h=blockIdx.z,split=blockIdx.x;','const int b=blockIdx.z,h=blockIdx.x,split=blockIdx.y;').replace('dim3 grid(S,qkv.size(0),hkv);','dim3 grid(hkv,S,qkv.size(0));').replace('muse_native_sliding_direct_v1','muse_native_sliding_head_neighbor_s30')
assert expected==v
# No Python plan/dispatch/helper changes, even nested functions.
a=[n for n in ast.walk(ast.parse(s)) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
b=[n for n in ast.walk(ast.parse(v)) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
assert [ast.dump(n) for n in a]==[ast.dump(n) for n in b]
def instructions(path):
    return re.findall(r"^\s*/\*[0-9a-f]+\*/\s+(.*?;)\s*/\*",path.read_text(),re.M)
base_sass=instructions(ROOT/'results/parent.sass.txt')
new_sass=instructions(ROOT/'results/neighbor.sass.txt')
assert len(base_sass)>4000 and len(base_sass)==len(new_sass)
assert sum(x!=y for x,y in zip(base_sass,new_sass))==12
mapping={'X':'Y','Y':'Z','Z':'X'}
assert [re.sub(r'SR_CTAID\.([XYZ])',lambda m:'SR_CTAID.'+mapping[m[1]],x) for x in base_sass]==new_sass
assert (ROOT/'results/parent.resources.txt').read_text()==(ROOT/'results/neighbor.resources.txt').read_text()
pairs=checks=0;summary={'sass_instruction_sites_all_specializations':len(base_sass),'sass_changed_coordinate_reads':12}
for file,count,rounds in [('initial',32,80),('confirmation',16,120)]:
    rows=json.loads((ROOT/f'results/{file}.json').read_text());assert len(rows)==count
    for row in rows:
        raw=row['raw'];assert len(raw)==rounds
        assert abs(row['paired_delta_us']-statistics.median(r[1]-r[0] for r in raw))<1e-9
        pairs+=len(raw)
    checks+=count//2*9
    summary[file]=[{k:r[k] for k in ('index','variant','parent_us','candidate_us','paired_delta_us')} for r in rows]
fresh=json.loads((ROOT/'results/fresh.json').read_text());assert len(fresh)==16 and all(r['replays']==4 for r in fresh)
log=(ROOT/'results/fresh.log').read_text();assert log.count("PASS ('fresh',")==64
suite=(ROOT/'results/suite.log').read_text();assert sum(line.startswith('PASS ') for line in suite.splitlines())==16 and 'FAIL' not in suite
assert sum('sglang on H200' in line for line in suite.splitlines())==12
profiles=json.loads((ROOT/'results/profile-summary.json').read_text());assert len(profiles)==6
for i in range(0,6,2):
    left,right=profiles[i:i+2]
    assert left['smsp__inst_executed.sum']==right['smsp__inst_executed.sum']
    assert abs(left['dram__bytes_read.sum']-right['dram__bytes_read.sum'])==256
summary['profile_reference_checks']=6
summary.update(production_sha256=hashlib.sha256(s.encode()).hexdigest(),candidate_sha256=hashlib.sha256(v.encode()).hexdigest(),unchanged_python_functions=len(a),source_edits=['native grid dimensions','native blockIdx mapping','extension name'],adjacent_pairs=pairs,comparison_reference_checks=checks,fresh_native_replays=64,full_suite_passes=16,bench_shapes=12,sanitizer_claim=False)
(ROOT/'results/audit.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ('initial','confirmation')},indent=2))

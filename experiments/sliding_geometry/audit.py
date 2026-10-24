"""CPU-only reproducibility and measurement-count audit for session 44."""
import ast
import hashlib
import json
import re
import statistics
from pathlib import Path
from generate import generate

ROOT = Path(__file__).resolve().parent
R = ROOT/'results'
previous = (ROOT/'previous.py').read_text()
candidate = (ROOT/'candidate.py').read_text()
unsigned = (ROOT/'unsigned.py').read_text()
assert candidate == generate(previous)
assert unsigned == candidate.replace('muse_native_sliding_geometry_s44', 'muse_native_sliding_geometry_unsigned_s44').replace(
    'const int tid=threadIdx.x,warp=tid/32,lane=tid%32;', 'const unsigned tid=threadIdx.x,warp=tid/32,lane=tid%32;').replace(
    'int i=tid+j*128,row=i/4,piece=(i%4)*16,t=lo+row;', 'unsigned i=tid+j*128,row=i/4,piece=(i%4)*16;int t=lo+row;')

def stripped(s):
    t = ast.parse(s)
    t.body = [n for n in t.body if not (isinstance(n,ast.Assign) and any(
        isinstance(v,ast.Name) and v.id=='_NATIVE_SLIDING' for v in n.targets))]
    return ast.dump(t)
assert stripped(previous) == stripped(candidate) == stripped(unsigned)
assert (ROOT.parents[1]/'submission.py').read_text() == previous

def assembly(file):
    out = {}
    for block in file.read_text().split('Function : ')[1:]:
        name = block.splitlines()[0]
        p = re.search(r'ILi(1|64)ELb([01])E(?:Li(0|2|16)E)?Ev', name)
        assert p, name
        key = (int(p[1]), int(p[2]), int(p[3] or 0))
        out[key] = re.findall(r'/\*[0-9a-f]+\*/\s+(.*?)\s*/\* 0x',block)
    return out

orig = assembly(R/'previous.sass.txt')
stats = {}
for label in ('candidate','unsigned'):
    asm = assembly(R/(label+'.sass.txt'))
    assert len(asm) == 12
    for key,ops in asm.items():
        assert sum('QGMMA' in x for x in ops) == 2
        assert sum('HGMMA' in x for x in ops) == 4
        assert sum('BAR.SYNC' in x for x in ops) == 3
        assert sum('LDGSTS' in x for x in ops) == 4
        if label == 'candidate' and key[2] == 0:
            assert ops == orig[key], ('generic changed',key)
    stats[label] = [dict(page=k[0],early=k[1],fixed_hkv=k[2],sites=len(v),
                        sites_before_first_copy=next(i for i,x in enumerate(v) if 'LDGSTS' in x))
                    for k,v in sorted(asm.items())]

# The unsigned copy topology enumerates exactly the same 256 distinct chunks.
chunks = []
for tid in range(128):
    for j in range(2):
        i = tid+j*128
        row,piece = i//4,(i%4)*16
        assert 0 <= row < 64 and piece in (0,16,32,48)
        chunks.append((row,piece))
assert len(set(chunks)) == 256

measurements = {}
for label in ('candidate','unsigned'):
    pairs = 0
    references = 0
    for mode,n in [('compare',14),('confirm',8)]:
        rows = json.loads((R/label/f'pairs-{mode}.json').read_text())
        assert len(rows) == n
        for row in rows:
            raw = row['raw']
            assert len(raw) == 80
            assert abs(statistics.median(b-a for a,b,e in raw)-row['paired_delta_us'])<1e-10
            e = statistics.median(x[2] for x in raw)
            assert abs(statistics.median(x[0] for x in raw)-e-row['previous_us'])<1e-10
            assert abs(statistics.median(x[1] for x in raw)-e-row['merged_us'])<1e-10
        pairs += n*80
        references += n*6
    fresh = json.loads((R/label/'fresh.json').read_text())
    assert len(fresh) == 14 and all(x['replays_per_source']==4 for x in fresh)
    fresh_log = (R/label/'fresh.log').read_text()
    assert fresh_log.count('PASS (\'fresh\'') == 112
    suite = (R/('suite.log' if label=='candidate' else 'unsigned-suite.log')).read_text()
    assert len(re.findall(r'^PASS \d+ ',suite,re.M)) == 16
    assert 'FAIL' not in suite
    assert len(re.findall(r'^\w+\s+\d+\s+(?:sliding|full)\s+',suite,re.M)) == 12
    measurements[label] = dict(adjacent_pairs=pairs,paired_reference_checks=references,
                               fresh_graph_replays=112,standard_tests=16,bench_shapes=12)

report = dict(source_sha256={k:hashlib.sha256(v.encode()).hexdigest() for k,v in
                            [('previous',previous),('candidate',candidate),('unsigned',unsigned)]},
              only_native_sliding_changed=True,generic_candidate_sass_identical=True,
              unchanged_copy_topology=True,assembly=stats,measurements=measurements)
(R/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

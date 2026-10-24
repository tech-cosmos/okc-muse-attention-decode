"""CPU-only frozen-source, counts, clock placement and complete graph audit."""
import ast,hashlib,json,subprocess,sys,statistics,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent
files=['production_snapshot.py','diagnostic.cu','binding.cpp']
before={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
subprocess.run([sys.executable,str(ROOT/'generate.py')],check=True)
assert before=={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}
assert (ROOT/'production_snapshot.py').read_bytes()==Path('submission.py').read_bytes()
assert before['production_snapshot.py']=='1bcf42aa6edf7f25a02117e3eabebb6f5b2e69967359137e015b2f79d445a548'
subprocess.run([sys.executable,str(ROOT/'analyze_sass.py')],check=True)
subprocess.run([sys.executable,str(ROOT/'summarize.py')],check=True,stdout=subprocess.DEVNULL)
subprocess.run([sys.executable,str(ROOT/'analyze_profiles.py')],check=True,stdout=subprocess.DEVNULL)
pairs=refs=samples=0
for name,count,rounds,nsamp in [('initial',7,40,12),('confirmation',5,80,20)]:
    rows=json.loads((ROOT/'results'/f'{name}.json').read_text());assert len(rows)==count
    log=(ROOT/'results'/f'{name}.log').read_text();assert log.count('PASS (')==count*15*3
    refs+=count*15*3
    for row in rows:
        assert len(row['timing'])==14 and len(row['cycles'])==12
        for mode,x in row['timing'].items():
            raw=x['raw'];assert len(raw)==rounds;pairs+=len(raw)
            assert abs(statistics.median(v[1]-v[0] for v in raw)-x['paired_delta_us'])<1e-9
        for mode,x in row['cycles'].items():
            assert len(x)==nsamp;samples+=len(x)
            for r in x:
                assert len(r)==row['shape']['batch']
                for b in r:
                    for h in b:
                        for s in h:assert len(s)==4 and all(t>=0 for t in s)
fresh=json.loads((ROOT/'results/fresh.json').read_text());assert len(fresh)==7
replays=sum(r['replays'] for r in fresh);assert replays==364
assert (ROOT/'results/fresh.log').read_text().count("PASS ('fresh'")==replays
suite=(ROOT/'results/suite.log').read_text();assert len(re.findall(r'^PASS \d+ ',suite,re.M))==16
assert len(re.findall(r'^(decode|verify) ',suite,re.M))==12
assert 'FAIL' not in suite and 'Traceback' not in suite
for name in ['parent-b1-p1','parent-b8-p1','parent-b1-p64']:
    assert 'PROFILE REFERENCE PASS' in (ROOT/'results'/f'ncu-{name}.log').read_text()
assert (ROOT/'results/profiles.log').read_text().count(' 0\n')==3
placements=json.loads((ROOT/'results/clock-placement.json').read_text());assert len(placements)==26
resources=(ROOT/'results/diagnostic.resources.txt').read_text();regs=list(map(int,re.findall(r'REG:(\d+)',resources)))
assert len(regs)==26 and min(regs)==74 and max(regs)==75
assert all(x=='0' for x in re.findall(r'(?:STACK|LOCAL):(\d+)',resources))
# Flat trace indexing is injective over every runtime trace shape used.
for case in fresh:
    shape=case['shape'];hk=16 if shape['heads']=='dp' else 2
    # S is <=32; enumerate the maximum and all smaller extents.
    for splits in range(1,33):
        offsets=[(((b*hk+h)*splits+s)*8+w*2+i) for b in range(shape['batch']) for h in range(hk) for s in range(splits) for w in range(4) for i in range(2)]
        assert sorted(offsets)==list(range(shape['batch']*hk*splits*8))
result=dict(production_sha256=before['production_snapshot.py'],generated_sources_reproducible=True,source_hashes=before,adjacent_graph_pairs=pairs,reference_checks_at_comparison_checkpoints=refs,independent_clock_trace_replays=samples,fresh_replays=replays,suite_passes=16,suite_actual_native_cases=3,bench_shapes=12,profile_reference_checks=3,unclocked_sass_identity_both_pages=True,clock_placement_variants=26,register_range=[min(regs),max(regs)],trace_index_injective=True,sanitizer='not run; no sanitizer coverage claimed',judge_requests=0)
assert pairs==9520 and refs==540 and samples==2208
(ROOT/'results/audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

"""CPU-only reproducibility, coverage and result audit."""
import hashlib,json,re,statistics,subprocess,sys
from pathlib import Path
P=Path(__file__).resolve().parent;R=P/'results'
files=['generate.py','tile256.cu','binding.cpp','cluster.cu','cluster_binding.cpp']
old={n:(P/n).read_bytes() for n in files}
for script in ['build_generator.py','generate.py','fuse.py']:
    subprocess.run([sys.executable,str(P/script)],check=True)
assert all((P/n).read_bytes()==s for n,s in old.items())
assert (P/'production_snapshot.py').read_bytes()==Path('submission.py').read_bytes()
# Every K/V gather has exactly the BN rows, four 16-byte pieces per row.
addresses=[(tid+j*128)//4*64+(tid+j*128)%4*16 for tid in range(128) for j in range(8)]
assert sorted(addresses)==list(range(0,256*64,16))
qkcols=[(i//8)*16+(lane%4)*2+(i&1)+(i&4)*2 for lane in range(4) for i in range(128) if not(i&2)]
assert sorted(qkcols)==list(range(256))
conversion=[8*tid+j*1024 for tid in range(128) for j in range(16)]
assert sorted(conversion)==list(range(0,256*64,8))
# Dynamic allocation: K 16384, Vraw 16384, Vhalf 32768. Reused only after PV wait + block barrier.
for name in ['tile256.cu','cluster.cu']:
    s=(P/name).read_text()
    assert 'storage+16384' in s and 'storage+32768' in s and '65536' in s
    assert s.count('m64n256k32.f32.e4m3.e4m3')==1
    assert 'unsigned pr[16][4]' in s and 'float score[128]' in s
s=(P/'cluster.cu').read_text()
assert s.count('cluster.sync();')==2
assert 'clusterDim={8,1,1}' in s and 'row=cluster.block_rank()' in s
assert 'cluster.map_shared_rank(partial,k)' in s
records=json.loads((R/'initial.json').read_text());assert len(records)==36
assert {r['index'] for r in records}==set(range(18))
for r in records:
    assert len(r['raw'])==60
    assert r['paired_delta_us']==statistics.median(x[1]-x[0] for x in r['raw'])
    assert r['paired_delta_us']>0
checks=(R/'initial.log').read_text().count('PASS (');assert checks==162
for variant in ['cluster','unfused']:
    fresh=json.loads((R/f'fresh-{variant}.json').read_text())
    assert len(fresh)==18 and sum(x['replays'] for x in fresh)==72
    assert (R/f'fresh-{variant}.log').read_text().count("PASS ('fresh',")==72
    assert len(re.findall(r'^PASS \d+', (R/f'suite-{variant}.log').read_text(),re.M))==16
for name in ['bench-cluster.log','suite-unfused.log']:
    t=(R/name).read_text();assert t.count('sglang on H200')==12
for tool in ['memcheck','synccheck','racecheck']:
    t=(R/f'{tool}.log').read_text()
    assert 'NATIVE_CLUSTER_REPLAYS 24' in t
    assert 'ERROR SUMMARY: 0 errors' in t or 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in t
resources={}
for name in ['cluster','unfused']:
    t=(R/f'{name}.resources.txt').read_text()
    rows=re.findall(r'REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)',t)
    assert len(rows)==4 and all(r[1]=='0' and r[3]=='0' for r in rows)
    resources[name]=rows
sass=(R/'cluster.sass.txt').read_text()
assert 'UCGABAR_ARV' in sass and 'UCGABAR_WAIT' in sass
out=dict(production_sha256=hashlib.sha256(Path('submission.py').read_bytes()).hexdigest(),
    source_sha256={n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in files},
    shape_count=18,paired_graphs=sum(len(r['raw']) for r in records),comparison_reference_checks=checks,
    fresh_replays=144,native_sanitizer_replays_per_tool=24,resources=resources,
    all_candidate_deltas_positive=True,production_unchanged=True,generator_reproducible=True)
(R/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

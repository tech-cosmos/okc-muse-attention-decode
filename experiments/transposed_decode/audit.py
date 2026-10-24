"""CPU-only reproduction and evidence audit; no CUDA or judge calls."""
import ast,csv,hashlib,json,re,statistics
from pathlib import Path
from generate import derive
R=Path(__file__).resolve().parent
root=R.parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(R/'production.py')=='62fefe79bd913e6dc1b12388ad7bec50139096cb07c7b61918b8031c0c93a86b'
assert sha(root/'submission.py')==sha(R/'production.py')
assert derive((R/'kernel.cu').read_text())==(R/'natural_v.cu').read_text()
# Matrix output fragments cover each (row,key or dimension; query head) once.
coords=[(w*16+l//4+r*8,l%4*2+c) for w in range(4) for l in range(32) for r in range(2) for c in range(2)]
assert len(coords)==len(set(coords))==64*8
# Gather copies exactly two contiguous 64-byte heads at each of 64 positions.
coords=[((t+j*128)//8,((t+j*128)%8)//4,((t+j*128)%4)*16+k) for t in range(128) for j in range(4) for k in range(16)]
assert len(coords)==len(set(coords))==64*2*64
for n in [512,4096]:
 for f in [lambda i:i^(((i>>7)&3)<<4),lambda i:i^(((i>>6)&7)<<3)]:
  assert sorted(f(i) for i in range(n))==list(range(n))
# Split partition covers all historical positions plus new token once, including empty contexts.
for seq in [0,1,10,63,64,65,257,4096,80000,1048576]:
 for s in [1,2,17,33,66,128]:
  chunk=(seq+1+64*s-1)//(64*s)*64
  spans=[(p*chunk,min(seq+1,(p+1)*chunk)) for p in range(s) if p*chunk<seq+1]
  assert spans[0][0]==0 and spans[-1][1]==seq+1
  assert all(spans[i][1]==spans[i+1][0] for i in range(len(spans)-1))
report=dict(production_sha256=sha(root/'submission.py'),sources={n:sha(R/n) for n in ['kernel.cu','natural_v.cu','native.py','generate.py']},variants={})
for mode in ['transpose','natural']:
 pairs=json.loads((R/f'{mode}-pairs.json').read_text());fresh=json.loads((R/f'{mode}-fresh.json').read_text())
 assert len(pairs)==len(fresh)==8
 for row in pairs:
  assert len(row['raw'])==60
  assert abs(statistics.median(x[1]-x[0] for x in row['raw'])-row['paired_delta_us'])<1e-9
 suite=(R/f'{mode}-suite.log').read_text();assert len(re.findall(r'^PASS \d+ ',suite,re.M))==16
 assert len(re.findall(r' us   \(sglang',suite))==12
 assert len(re.findall(r'^PASS ',(R/f'{mode}-compare.log').read_text(),re.M))==48
 assert len(re.findall(r'^PASS ',(R/f'{mode}-fresh.log').read_text(),re.M))==64
 smoke=[json.loads(l) for l in (R/f'{mode}-smoke.log').read_text().splitlines() if l.startswith('{')]
 assert len(smoke)==16 and all(x['ok'] for x in smoke)
 sass=(R/f'{mode}-sass.txt').read_text()
 counts={op:len(re.findall(r'\b'+op+r'\b',sass)) for op in ['QGMMA','HGMMA','F2FP','STS','SHFL','BAR','LDGSTS']}
 assert counts['QGMMA']==8 and counts['HGMMA']==16
 report['variants'][mode]=dict(graph_pairs=480,comparison_checks=48,fresh_native_replays=32,fresh_parent_replays=32,smoke_checks=16,suite_passes=16,sass_sites_all_two_pages=counts,pairs=[{k:v for k,v in r.items() if k!='raw'} for r in pairs])
for tool in ['memcheck','synccheck','racecheck']:
 s=(R/f'natural-{tool}.log').read_text();assert 'TOTAL 24' in s
 assert len(re.findall(r'^PASS ',s,re.M))==24
 if tool=='racecheck':assert 'RACECHECK SUMMARY: 0 hazards' in s
 else:assert 'ERROR SUMMARY: 0 errors' in s
for mode in ['transpose','natural']:
 s=(R/f'{mode}-ncu.log').read_text();assert f'PASS profile {mode}' in s
 assert 'smsp__inst_executed.sum' in s
 lines=s.splitlines();start=next(i for i,l in enumerate(lines) if l.startswith('"ID"'))
 report['variants'][mode]['ncu_metrics']={row['Metric Name']:float(row['Metric Value'].replace(',','')) for row in csv.DictReader(lines[start:])}
report['sanitizer_fresh_replays_per_tool']=24
# CUDA Shared fields aligned to 1024 bytes as a structure.
fields=[(512,3*2*4096),(16,3*2*4096),(1024,2*4096*2),(512,2*512),(1024,2*512*2),(1,2*128),(4,2*4*8*4)]
offset=0
for align,size in fields:offset=(offset+align-1)//align*align+size
dynamic=(offset+1023)//1024*1024
assert dynamic==69632
report['dynamic_shared_bytes']=dynamic
(R/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

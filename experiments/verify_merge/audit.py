from pathlib import Path
import hashlib,json,re
root=Path(__file__).resolve().parent
results=root/'results'
report={'production_sha256':hashlib.sha256(Path('submission.py').read_bytes()).hexdigest()}
assert report['production_sha256']=='6b171beb8ed5e2a306ff67c3820ba7e83943f73c1a73d7b06eeeb630e0d0eecf'
diag=json.loads((results/'diagnostic-103.json').read_text())
assert len(diag)==12 and all(len(r['raw'])==80 for r in diag)
report['diagnostic_entries']=len(diag);report['diagnostic_pairs_per_entry']=80
report['comparisons']={}
for variant in ('warp','striped'):
 rows=json.loads((results/f'pairs-{variant}-False-103-False.json').read_text())
 assert len(rows)==5 and all(len(r['raw'])==60 for r in rows)
 report['comparisons'][variant]=[{k:v for k,v in r.items() if k!='raw'} for r in rows]
 log=(results/f'validate-{variant}.log').read_text()
 assert len(re.findall(r'^PASS \d+ ',log,re.M))==16
 assert len(re.findall(r'^PASS native fresh ',log,re.M))==7
 assert '42 native fresh graph replays PASS' in log
 bench=re.search(r'H100 time vs SGLang-on-H200, geomean over loads: ([\d.]+)',log)
 assert bench
 report[variant]={'standard_fallback_passes':16,'native_fresh_replays':42,'local_bench_geomean':float(bench[1])}
assert 'PASS total 48' in (results/'synthetic.log').read_text()
report['synthetic_replays']=48
sass=json.loads((results/'sass-audit.json').read_text());assert len(sass)==6 and all(r['coherent_loads_after_wait'] for r in sass.values())
report['sass_specializations_checked']=6
report['sanitizers']={}
for tool in ('memcheck','synccheck','racecheck'):
 log=(results/f'{tool}.log').read_text()
 assert 'PASS total 48' in log
 if tool=='racecheck':assert 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in log
 else:assert 'ERROR SUMMARY: 0 errors' in log
 report['sanitizers'][tool]={'synthetic_native_replays':48,'errors':0}
report['source_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.suffix in ('.py','.cu')}
(results/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('comparisons','source_hashes')},indent=2))

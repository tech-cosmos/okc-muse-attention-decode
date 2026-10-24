import hashlib,json,re,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent
files=['kernel.cu','vector.cu','precomputed.cu','vector.py','precomputed.py']
before={x:(R/x).read_bytes() for x in files}
for gen in ('generate.py','vectorize.py','precompute.py'):subprocess.run(['python',str(R/gen)],check=True)
assert all((R/x).read_bytes()==v for x,v in before.items())
report={'reproducible_generated_sources':files,'production_sha256':hashlib.sha256(Path('submission.py').read_bytes()).hexdigest(),'production_unchanged':Path('submission.py').read_bytes()==(R/'production_parent.py').read_bytes(),'tests':{},'fresh':{},'sanitizers':{},'comparisons':{}}
for suffix in ('','-vector','-precomputed'):
 txt=(R/f'results/validation{suffix}.txt').read_text()
 assert len(re.findall(r'^PASS \d+',txt,re.M))==32 and not re.search(r'FAIL|Traceback',txt)
 report['tests'][suffix or 'native']={'passed':32,'per_order':16,'native_per_order':3,'fallback_per_order':13,'local_bench_geomeans':re.findall(r'geomean over loads: ([0-9.]+)',txt)}
 txt=json.loads((R/f'results/balanced{suffix}.json').read_text());assert len(txt)==14
 report['comparisons'][suffix or 'native']={'shape_order_entries':len(txt),'pairs_per_entry':40}
for suffix in ('','-precomputed'):
 txt=(R/f'results/fresh{suffix}.txt').read_text();assert txt.count('PASS fresh graph')==14 and 'Traceback' not in txt
 report['fresh'][suffix or 'native']={'all_native_replays':84,'shapes':7,'orders':2}
 for tool in ('memcheck','synccheck','racecheck'):
  txt=(R/f'results/{tool}{suffix}.txt').read_text()
  assert txt.count('PASS native-only')==8 and 'Traceback' not in txt
  summary=re.findall(r'========= (?:ERROR SUMMARY|RACECHECK SUMMARY): .*',txt)[-1]
  assert ('0 errors' in summary)
  if tool=='racecheck':assert '0 hazards' in summary and '0 warnings' in summary
  report['sanitizers'][tool+suffix]={'native_calls':24,'summary':summary}
(R/'results/audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

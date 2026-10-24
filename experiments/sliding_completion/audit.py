import hashlib,json,re,subprocess,sys
from pathlib import Path
p=Path(__file__).resolve().parent;r=p/'results';report={}
for stem in('kernel','prefetched'):
 before=(p/(stem+'.cu')).read_bytes()
 subprocess.run([sys.executable,str(p/('generate.py' if stem=='kernel' else 'prefetch.py'))],check=True)
 assert before==(p/(stem+'.cu')).read_bytes()
 report[stem+'_sha256']=hashlib.sha256(before).hexdigest()
for suffix in('','-prefetched'):
 val=(r/('validation'+suffix+'.txt')).read_text();assert len(re.findall(r'^PASS \d+ ',val,re.M))==16 and 'FAIL' not in val
 fresh=(r/('fresh'+suffix+'.txt')).read_text();assert fresh.count('PASS fresh graph')==8
 cmp=json.loads((r/('comparison'+suffix+'.json')).read_text());assert len(cmp)==16 and all(len(c['raw'])==40 for c in cmp)
 for tool in('memcheck','synccheck','racecheck'):
  s=(r/(tool+suffix+'.txt')).read_text();assert s.count('PASS native-only')==8
  assert ('0 hazards displayed (0 errors, 0 warnings)' if tool=='racecheck' else 'ERROR SUMMARY: 0 errors') in s
 report[suffix or 'original']={'tests':16,'native_tests':3,'fresh_native_replays':48,'comparison_entries':16,'pairs_per_entry':40,'sanitizer_calls_per_tool':24}
d=json.loads((r/'diagnostic.json').read_text());assert len(d)==4 and all(len(x['raw'])==60 for x in d)
report['diagnostic_entries']=4
prod=hashlib.sha256(Path('submission.py').read_bytes()).hexdigest();assert prod=='16a7db558d53ddc110e2bf6ccec1ff243f4407b817ec939eeee409fbe2b8f16e';report['unchanged_production_sha256']=prod
b=(r/'bench.txt').read_text();assert b.count('geomean over loads:')==2;report['full_bench_geomeans']=re.findall(r'geomean over loads: ([0-9.]+)',b)
(r/'audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

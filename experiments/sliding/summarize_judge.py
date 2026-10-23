"""Summarize normal okc status JSON; never imported by a submission."""
import json,sys
from pathlib import Path
for name in sys.argv[1:]:
 x=json.loads(Path(name).read_text());r=x.get('result')
 print(x['id'],x['status'])
 if not r:continue
 print('passed:',r.get('passed'),'speedup:',r.get('speedup'),'95% interval:',r.get('speedup_ci'),'error:',r.get('error'))
 print('| load | layer | candidate us | baseline us | speedup |')
 print('|---|---|---:|---:|---:|')
 for rec in r.get('records',[]):
  if rec.get('event')=='benchmark':
   c=rec['case'];print(f"| {c['load']} | {c['layer']} | {rec['median_ns']/1000:.3f} | {rec['baseline_ns']/1000:.3f} | {rec['speedup']:.3f} |")

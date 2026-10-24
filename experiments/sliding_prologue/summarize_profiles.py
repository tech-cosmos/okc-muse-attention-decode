import csv,io,json
from pathlib import Path
root=Path(__file__).resolve().parent/'results';report={}
for name in ('base-early','direct-late','direct-early'):
 lines=(root/f'profile-{name}.csv').read_text().splitlines();start=next(i for i,s in enumerate(lines) if s.startswith('"Address","Source"'))
 rows=list(csv.DictReader(io.StringIO('\n'.join(lines[start:]))))
 def num(s):
  try:return float(s.replace(',',''))
  except (ValueError,AttributeError):return 0.
 totals={k:sum(num(r.get(k)) for r in rows) for k in rows[0] if k.startswith('stall_') and 'Not Issued' not in k}
 tops=sorted(rows,key=lambda r:num(r.get('# Samples')),reverse=True)[:12]
 report[name]={'instructions_executed':sum(num(r.get('Instructions Executed')) for r in rows),'thread_instructions':sum(num(r.get('Thread Instructions Executed')) for r in rows),'shared_excess_wavefronts':sum(num(r.get('L1 Wavefronts Shared Excessive')) for r in rows),'stall_totals':totals,'top_stalls':[{'source':r['Source'].strip(),'samples':num(r.get('# Samples')),'reason':max(totals,key=lambda k:num(r.get(k)))} for r in tops]}
 print(name,json.dumps(report[name]),flush=True)
(root/'profile-summary.json').write_text(json.dumps(report,indent=2)+'\n')

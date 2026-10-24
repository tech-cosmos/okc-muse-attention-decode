"""Summarize archived results; fail if promised validation is incomplete."""
from pathlib import Path
import json,re
p=Path(__file__).resolve().parent;r=p/'results'
variants=('overlap','latebar','nobar','latebar_nobar')
summary={'tests':{},'bench':{},'fresh':{},'sanitizers':{},'comparison':[],'profiles':{}}
for v in variants:
 text=(r/('test-'+v+'.log')).read_text()
 n=len(re.findall(r'^PASS ',text,re.M));assert n==16,(v,n)
 assert 'suite passed; native cases 5' in text
 summary['tests'][v]={'passed':n,'native':5,'fallback':11}
 text=(r/('bench-'+v+'.log')).read_text()
 m=re.search(r'geomean over loads: ([\d.]+)',text);assert m,v
 summary['bench'][v]=float(m[1])
for v in ('nobar','latebar_nobar'):
 text=(r/('fresh-'+v+'.log')).read_text()
 n=len(re.findall('PASS fresh graph',text));assert n==7,(v,n)
 summary['fresh'][v]={'shapes':n,'replays':n*6}
for tool in ('memcheck','synccheck','racecheck'):
 text=(r/(tool+'.log')).read_text()
 assert len(re.findall('PASS native-only reference',text))==8,tool
 assert ('ERROR SUMMARY: 0 errors' in text or 'RACECHECK SUMMARY: 0 hazards' in text),tool
 summary['sanitizers'][tool]=[l for l in text.splitlines() if 'SUMMARY:' in l]
for row in json.loads((r/'compare-final.json').read_text()):
 summary['comparison'].append({k:v for k,v in row.items() if k!='raw_us'})
for v in ('baseline','nobar','latebar_nobar'):
 text=(r/('profile-'+v+'.log')).read_text()
 summary['profiles'][v]=[l.strip() for l in text.splitlines() if any(term in l for term in ('Achieved Occupancy','Issued Warp Per Scheduler','Warp Cycles Per Issued Instruction','bank_conflicts','Theoretical Active Warps per SM'))]
summary['source_wavefronts']={}
for v in ('baseline','nobar','latebar_nobar'):
 rows=json.loads((r/('source-'+v+'-summary.json')).read_text())
 summary['source_wavefronts'][v]={}
 for group,term in [('raw_v','LDS.64'),('gathers','LDGSTS.E.'),('qk','QGMMA.')]:
  selected=[row for row in rows if term in row['Source']]
  summary['source_wavefronts'][v][group]={col:sum(int(float(row[col] or 0)) for row in selected) for col in ('L1 Wavefronts Shared','L1 Wavefronts Shared Ideal','L1 Wavefronts Shared Excessive')}
(p/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))

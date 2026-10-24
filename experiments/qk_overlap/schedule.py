"""Static stage4/page1 instruction ordering around QK, V staging, and waits."""
from pathlib import Path
import json,re
p=Path(__file__).resolve().parent
summary={}
for name in ('baseline','overlap','latebar','nobar','latebar_nobar'):
 lines=(p/(name+'.sass')).read_text().splitlines()
 selected=[]
 for line in lines:
  if any(op in line for op in ('QGMMA.','HGMMA.','WARPGROUP.DEPBAR','WARPGROUP.ARRIVE','STS.128','LDS.64','F2FP.F16.E4M3','BAR.SYNC','FENCE.VIEW')):
   selected.append(line.split(';')[0].strip()+';')
 summary[name]=selected
(p/'schedule-summary.json').write_text(json.dumps(summary,indent=2)+'\n')

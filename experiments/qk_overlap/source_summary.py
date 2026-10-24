"""Summarize NCU instruction-attributed shared wavefronts and stall samples."""
import csv,json,sys
from pathlib import Path
path=Path(sys.argv[1]);rows=list(csv.reader(path.open()))
k=next(i for i,r in enumerate(rows) if r and r[0]=='Address')
h=rows[k];data=[dict(zip(h,r)) for r in rows[k+1:] if len(r)==len(h)]
result=[]
for r in data:
 if any(x in r['Source'] for x in ('LDS.64','LDGSTS.E.','QGMMA.','HGMMA.','STS.128')) and float(r['Instructions Executed'] or 0)>0:
  result.append({k:r[k] for k in ('Address','Source','Instructions Executed','L1 Conflicts Shared N-Way','L1 Wavefronts Shared Excessive','L1 Wavefronts Shared','L1 Wavefronts Shared Ideal','Warp Stall Sampling (All Samples)')})
print(json.dumps(result,indent=2))

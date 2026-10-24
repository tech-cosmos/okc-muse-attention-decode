"""Static instruction counts only; not executed-instruction or latency counts."""
from pathlib import Path
import re,json,subprocess
root=Path(__file__).resolve().parent;r={}
for name in ('base','pruned','half'):
 so=root/('.build-'+name)/('codex_sliding_'+('base' if name=='base' else 'pruned' if name=='pruned' else 'half')+'_rows.so')
 for flag,out in (('-sass',name+'.sass'),('-res-usage',name+'-resources.txt')):
  (root/'results'/out).write_text(subprocess.check_output(['/usr/local/cuda-13.0/bin/cuobjdump',flag,str(so)],text=True))
 body=next(b for b in (root/'results'/(name+'.sass')).read_text().split('Function : ') if b.startswith('_Z7slidingILi1ELb1EE'))
 ops=re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?(\w+(?:\.\w+)*)',body)
 r[name]={'instructions':len(ops),'counts':{k:sum(o.startswith(k) for o in ops) for k in ['MUFU.EX2','MUFU.RSQ','QGMMA','HGMMA','F2FP','SHFL','BAR','BRA','BSSY','BSYNC']}}
(root/'results'/'sass-summary.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))

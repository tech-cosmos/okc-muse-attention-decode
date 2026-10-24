"""Static instruction/resource evidence; no GPU launch or profiler timings."""
from pathlib import Path
import re,json,subprocess
root=Path(__file__).resolve().parent
report={}
paths={'base':Path.home()/'.cache/torch_extensions/py312_cu130/muse_native_sliding_direct_v1/muse_native_sliding_direct_v1.so'}
paths.update({n:root/('.build-'+n)/('codex_sliding_qk_'+n+'.so') for n in ('overlap','late','pinned')})
for name,so in paths.items():
 for flag,out in (('-sass',name+'.sass'),('-res-usage',name+'-resources.txt')):
  (root/'results'/out).write_text(subprocess.check_output(['/usr/local/cuda-13.0/bin/cuobjdump',flag,str(so)],text=True))
 body=next(b for b in (root/'results'/(name+'.sass')).read_text().split('Function : ') if b.startswith('_Z7slidingILi1ELb1EE'))
 ops=re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?(\w+(?:\.\w+)*)',body)
 report[name]={'instructions':len(ops),'counts':{k:sum(o.startswith(k) for o in ops) for k in ['MUFU.EX2','MUFU.RSQ','QGMMA','HGMMA','HMMA','LDSM','F2FP','SHFL','BAR','BRA','BSSY','BSYNC']}}
(root/'results'/'sass-summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

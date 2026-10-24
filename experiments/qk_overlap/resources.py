"""Archive compiler resource metadata for stage4/page1; no GPU work."""
import subprocess,re,json
from pathlib import Path
p=Path(__file__).resolve().parent
res={}
for name in ('baseline','overlap','latebar','nobar','latebar_nobar'):
 so=next((p/('.build-'+name)).glob('*.so'))
 s=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(so)],text=True)
 match=re.search(r'Function _Z6verifyILi4ELi1EE.*?(?=\n Function |\Z)',s,re.S)
 assert match,name
 res[name]=match.group().strip()
(p/'resources.json').write_text(json.dumps(res,indent=2)+'\n')
print(json.dumps(res,indent=2))

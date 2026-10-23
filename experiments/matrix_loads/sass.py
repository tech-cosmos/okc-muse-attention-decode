"""Archive only stage4/page1 native SASS and static operation counts."""
import subprocess,re,json
from collections import Counter
from pathlib import Path
p=Path(__file__).resolve().parent
results={}
for name in ('baseline','x4','firstwrite','combined','vector','all','forced'):
    build=p/('.build-pair-mbar' if name=='baseline' else '.build-'+name)
    lib=next(build.glob('*.so'))
    text=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(lib)],text=True)
    match=re.search(r'Function : _Z6verifyILi4ELi1EE.*?(?=\n\s*Function :|\Z)',text,re.S)
    assert match,name
    (p/(name+'.sass')).write_text(match.group())
    counts=Counter()
    for line in match.group().splitlines():
        m=re.search(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)\b',line)
        if m:counts[m[1]]+=1
    results[name]=dict(counts)
(p/'sass-counts.json').write_text(json.dumps(results,indent=2)+'\n')
for name,c in results.items():
    print(name,{k:v for k,v in c.items() if k.startswith(('LDS','STS','HGMMA','CS2R','F2FP'))})

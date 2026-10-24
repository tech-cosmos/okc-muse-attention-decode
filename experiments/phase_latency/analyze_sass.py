"""Check actual interval placements and exact no-clock SASS control identity."""
import re,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def functions(path):
    s=Path(path).read_text();out={}
    for part in re.split(r'\s*Function : ',s)[1:]:
        name=part.splitlines()[0];m=re.search(r'ILi(\d+)ELb(\d)(?:ELi(n?\d+))?',name);page,early,phase=m.groups()
        inst=[(addr,re.sub(r'\s+',' ',op.strip())) for addr,op in re.findall(r'/\*([0-9a-f]+)\*/\s+(.*?)\s*;\s*/\*',part)]
        out[(int(page),int(early),None if phase is None else int(phase.replace('n','-')))]=inst
    return out
p=functions(ROOT/'results/parent.sass.txt');c=functions(ROOT/'results/diagnostic.sass.txt');out=[]
for page in (1,64):
    assert [x[1] for x in p[page,1,None]]==[x[1] for x in c[page,1,-1]]
for (page,early,phase),inst in c.items():
    clocks=[i for i,(_,op) in enumerate(inst) if 'SR_CLOCKLO' in op]
    assert len(clocks)==(0 if phase==-1 else 2)
    ops=[op for _,op in inst[clocks[0]+1:clocks[1]]] if clocks else []
    if phase==11:assert ops==[]
    if phase==2:assert len(ops)==1 and 'DEPBAR.LE SB0' in ops[0]
    if phase==6:assert sum('QGMMA' in op for op in ops)==2 and sum('WARPGROUP.DEPBAR' in op for op in ops)==1
    if phase==8:assert sum('HGMMA' in op for op in ops)==4 and sum('WARPGROUP.DEPBAR' in op for op in ops)==1
    out.append(dict(page=page,phase=phase,clock_offsets=[inst[i][0] for i in clocks],between=ops))
(ROOT/'results/clock-placement.json').write_text(json.dumps(out,indent=2)+'\n')
print('No-clock SASS identical:',{page:len(p[page,1,None]) for page in (1,64)},'26 clock placements pass')

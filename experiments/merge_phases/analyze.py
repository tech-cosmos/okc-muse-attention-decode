"""CPU-only disassembly checks; intervals describe emitted instruction windows."""
from pathlib import Path
import re,json
R=Path(__file__).resolve().parent
def functions(file):
    s=(R/file).read_text();parts=re.split(r'Function : (\S+)\n',s);out={}
    for name,body in zip(parts[1::2],parts[2::2]):
        lines=[]
        for line in body.splitlines():
            m=re.search(r'/\*[0-9a-f]+\*/\s+(.*?)\s*;\s*/\*',line)
            if m:lines.append(m.group(1)+';')
        out[name]=lines
    return out
old=functions('previous.sass');new=functions('candidate.sass');summary=[]
for name,lines in new.items():
    m=re.search(r'ILi8ELi16ELb([01])ELi(\d+)E',name);assert m,name
    pdl,phase=map(int,m.groups());clocks=[i for i,x in enumerate(lines) if 'CS2R' in x and 'SR_CLOCKLO' in x]
    waits=[i for i,x in enumerate(lines) if 'ACQBULK' in x]
    loads=[i for i,x in enumerate(lines) if 'LDG' in x and 'STRONG.GPU' in x]
    assert all(not pdl or i>waits[0] for i in loads)
    if phase==0:
        parent=next(v for k,v in old.items() if f'ELb{pdl}EE' in k)
        assert lines==parent,(name,'no-clock code changed')
    else: assert len(clocks)==2,(name,clocks)
    summary.append(dict(name=name,pdl=bool(pdl),phase=phase,instructions=len(lines),clocks=clocks,waits=waits,coherent_loads=len(loads),clock_contexts=[lines[max(0,i-4):i+5] for i in clocks],between_clocks=lines[clocks[0]+1:clocks[1]] if clocks else []))
assert len(summary)==20
(R/'sass-analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps([{k:v for k,v in x.items() if k not in ('clock_contexts','between_clocks')} for x in summary],indent=2))

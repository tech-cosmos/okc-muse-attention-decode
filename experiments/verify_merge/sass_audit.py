from pathlib import Path
import re,json
root=Path(__file__).resolve().parent
s=(root/'results/sass.txt').read_text();out={}
for text in s.split('Function : ')[1:]:
 name=text.splitlines()[0].strip();lines=[x for x in text.splitlines() if re.search(r'/\*[0-9a-f]+\*/',x)]
 acq=next((i for i,x in enumerate(lines) if 'ACQBULK' in x),None)
 cg=[i for i,x in enumerate(lines) if 'LDG' in x and 'STRONG.GPU' in x]
 out[name]={'instruction_lines':len(lines),'wait_index':acq,'coherent_load_sites':len(cg),'coherent_loads_after_wait':all(i>acq for i in cg) if acq is not None else False,'load_examples':[lines[i].strip() for i in cg[:3]]}
assert len(out)==6 and all(v['coherent_loads_after_wait'] and v['coherent_load_sites'] for v in out.values())
(root/'results/sass-audit.json').write_text(json.dumps(out,indent=2)+'\n')
print('PASS six compiled specializations: coherent partial load sites after dependency wait')

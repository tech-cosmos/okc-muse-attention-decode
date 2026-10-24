import csv,io,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
rows=[]
for b,p in [(1,1),(8,1),(8,64)]:
    for name in ['parent','neighbor']:
        text=(ROOT/'results'/f'ncu-{name}-b{b}-p{p}.log').read_text()
        assert "PASS ('profile'," in text and '==ERROR==' not in text
        data=list(csv.DictReader(io.StringIO(text[text.index('"ID",'):])));assert len(data)==5
        row=dict(batch=b,page=p,variant=name,grid=data[0]['Grid Size'])
        row.update({r['Metric Name']:float(r['Metric Value'].replace(',','')) for r in data});rows.append(row)
(ROOT/'results/profile-summary.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))

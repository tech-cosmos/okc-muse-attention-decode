"""Export readable ranges and kernel intervals from nsys SQLite output."""
import json
import sqlite3
import statistics
from pathlib import Path
root=Path(__file__).resolve().parent/'results'
c=sqlite3.connect(root/'timeline.sqlite');c.row_factory=sqlite3.Row
ranges=c.execute("select start,end,text from NVTX_EVENTS where text like 'case%:repeat%' order by start").fetchall()
result=[]
for span in ranges:
    rows=c.execute('select k.*,s.value as name from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id where k.start>=? and k.end<=? order by k.start',(span['start'],span['end'])).fetchall()
    assert len(rows)==2,(span['text'],len(rows))
    att,merge=rows;assert att['name'].startswith('_attn_split')
    assert merge['name'].startswith('_combine') or 'reduce_kernel' in merge['name']
    record=dict(label=span['text'],cpu_start_ns=span['start'],cpu_end_ns=span['end'],
                attention_us=(att['end']-att['start'])/1000,merge_us=(merge['end']-merge['start'])/1000,
                merge_start_after_att_start_us=(merge['start']-att['start'])/1000,
                kernel_overlap_us=max(0,min(att['end'],merge['end'])-max(att['start'],merge['start']))/1000,
                kernels=[dict(r) for r in rows])
    result.append(record)
assert len(result)==24
(root/'timeline-intervals.json').write_text(json.dumps(result,indent=2)+'\n')
summary=[]
for case in ('case3','case13'):
    for name in ('main','native','serial_main','serial_native'):
        rows=[r for r in result if r['label'].startswith(case+':'+name+':')]
        assert len(rows)==3
        s=dict(case=case,mode=name)
        for key in ('attention_us','merge_us','merge_start_after_att_start_us','kernel_overlap_us'):
            s[key]=statistics.median(r[key] for r in rows)
        summary.append(s);print(s)
(root/'timeline-summary.json').write_text(json.dumps(summary,indent=2)+'\n')

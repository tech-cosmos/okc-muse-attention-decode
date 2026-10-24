import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def number(x):
    try:return float(x.replace(',',''))
    except (ValueError,AttributeError):return None
results=[]
for name in ['parent-b1-p1','parent-b8-p1','parent-b1-p64']:
    lines=(ROOT/'results'/f'ncu-{name}-source.csv').read_text().splitlines()
    idx=next(i for i,l in enumerate(lines) if l.startswith('"Address"'))
    rows=list(csv.DictReader(lines[idx:]));rows=[r for r in rows if r.get('Address','').startswith('0x')]
    stalls={k:sum(number(r[k]) or 0 for r in rows) for k in rows[0] if k.startswith('stall_') and 'Not Issued' not in k}
    tops=sorted(rows,key=lambda r:number(r.get('stall_long_sb')) or 0,reverse=True)[:8]
    with (ROOT/'results'/f'ncu-{name}-raw.csv').open() as f:raw=list(csv.DictReader(f))[-1]
    metrics={k:v for k,v in raw.items() if k and (k.startswith('smsp__average_warp_latency') or k.startswith('smsp__warp_issue_stalled') or k.startswith('sm__warps_active') or k in ['launch__registers_per_thread','launch__occupancy_limit_registers','smsp__inst_executed.sum','smsp__pcsamp_sample_count'])}
    results.append(dict(name=name,all_sample_stalls=stalls,metrics=metrics,top_long_scoreboard_sites=[{k:r[k] for k in ['Address','Source','stall_long_sb','# Samples']} for r in tops]))
(ROOT/'results/profile-summary.json').write_text(json.dumps(results,indent=2)+'\n')
for r in results:
    print(r['name'],r['all_sample_stalls'])
    print(r['metrics']);print(r['top_long_scoreboard_sites'][:3])

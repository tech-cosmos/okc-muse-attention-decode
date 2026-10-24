"""Summaries retain raw samples and do not add nonidentical compiled intervals."""
import json,statistics,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent
names=['metadata_and_gather_issue','query_newkv_preparation','residual_copy_wait','first_CTA_rendezvous','newkv_publish','value_conversion_publish','QK','softmax_pack','PV','partial_output','total','clock_pair_overhead']
results={}
for file in ['initial','confirmation']:
    rows=[]
    for x in json.loads((ROOT/'results'/f'{file}.json').read_text()):
        stages={}
        for k,v in x['cycles'].items():
            allrows=[a for r in v for b in r for h in b for a in h]
            # Equal CTA weighting within each launch; warp0 carries all real output rows.
            warpmed=[statistics.median(row[w] for row in allrows) for w in range(4)]
            owners=[];ordinary=[]
            for r in v:
                for bi,b in enumerate(r):
                    seq=x['seq_lens'][bi];base=max(0,seq-2047);owner=(seq-base)//64
                    for h in b:
                        for si,a in enumerate(h):(owners if si==owner else ordinary).append(a)
            stages[names[int(k)]]=dict(warp_median_cycles=warpmed,owner_warp_medians=[statistics.median(a[w] for a in owners) for w in range(4)],ordinary_warp_medians=[statistics.median(a[w] for a in ordinary) for w in range(4)] if ordinary else None)
        rows.append(dict(index=x['index'],shape=x['shape'],stages=stages,timing={k:{a:b for a,b in v.items() if a!='raw'} for k,v in x['timing'].items()}))
    results[file]=rows
(ROOT/'results/summary.json').write_text(json.dumps(results,indent=2)+'\n')
for x in results['confirmation']:
 print(x['index'],[(k, v['warp_median_cycles'][0]) for k,v in x['stages'].items()])

import csv
import io
import json
from pathlib import Path

root = Path(__file__).resolve().parent / 'results'
summary = {}
for name in ('base', 'query', 'both'):
    text = (root / f'profile-{name}.csv').read_text()
    rows = list(csv.DictReader(io.StringIO(text[text.index('"Address",'):])) )
    def count(key, records):
        return sum(int(r[key].replace(',', '')) for r in records
                   if r.get(key, '').replace(',', '').isdigit())
    groups = {
        'fp32_to_fp8': [r for r in rows if 'F2FP.SATFINITE.E4M3.F32' in r['Source']],
        'byte_permutation': [r for r in rows if 'PRMT' in r['Source']],
        'all': rows,
    }
    summary[name] = {
        key: {'sites': len(rs), 'executed_warp_instructions': count('Instructions Executed', rs)}
        for key, rs in groups.items()
    }
    summary[name]['shared_excess_wavefronts'] = count('L1 Wavefronts Shared Excessive', rows)
(root / 'profile-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))

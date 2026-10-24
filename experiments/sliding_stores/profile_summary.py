import csv
import io
import json
from pathlib import Path

R = Path(__file__).resolve().parent / 'results'
summary = {}
for name in ('base', 'explicit'):
    s = (R / ('profile-' + name + '.csv')).read_text()
    rows = list(csv.DictReader(io.StringIO(s[s.index('"Address",'):])))
    stores = [r for r in rows if 'STG.E ' in r['Source'] or 'STG.E.64 ' in r['Source']]
    def total(key, records):
        return sum(int(r[key].replace(',', '')) for r in records if r.get(key, '').replace(',', '').isdigit())
    summary[name] = {
        'partial_store_sites': len(stores),
        'partial_store_executed_warp_instructions': total('Instructions Executed', stores),
        'partial_store_l2_theoretical_sectors': total('L2 Theoretical Sectors Global', stores),
        'partial_store_l2_theoretical_excessive_sectors': total('L2 Theoretical Sectors Global Excessive', stores),
        'all_executed_warp_instructions': total('Instructions Executed', rows),
        'all_shared_excessive_wavefronts': total('L1 Wavefronts Shared Excessive', rows),
    }
(R / 'profile-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))

"""Audit the exact integrated source and its independently collected evidence."""
import ast
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
candidate = (ROOT / 'submission.py').read_bytes()
previous = (ROOT / 'previous.py').read_bytes()
main = (ROOT / 'main.py').read_bytes()
assert candidate == subprocess.check_output(['git', 'show', 'd125df9:submission.py'])
assert hashlib.sha256(candidate).hexdigest() == 'ed5bde6e29882a90249d327c30602a9b07e83ad0d710288420c7a6c7eb23f9ca'
assert previous == subprocess.check_output(['git', 'show', 'a2a18e0:submission.py'])
assert main == subprocess.check_output(['git', 'show', '06b7755:submission.py'])

def functions(source):
    return {n.name: ast.dump(n, include_attributes=False) for n in ast.walk(ast.parse(source))
            if isinstance(n, ast.FunctionDef)}

cf, mf = functions(candidate), functions(main)
assert set(cf) == set(mf)
helpers = [name for name in mf if name != 'kernel_fn']
assert all(cf[name] == mf[name] for name in helpers)

def assigns(source):
    return {n.targets[0].id: ast.dump(n.value, include_attributes=False)
            for n in ast.parse(source).body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}

ca, pa, ma = assigns(candidate), assigns(previous), assigns(main)
for name in ('_NATIVE_MERGE', '_NATIVE_SLIDING'):
    assert ca[name] == pa[name]
assert ca['_PDT'] == ma['_PDT']
assert cf['kernel_fn'] == functions(previous)['kernel_fn']

report = dict(candidate_sha256=hashlib.sha256(candidate).hexdigest(),
              previous_sha256=hashlib.sha256(previous).hexdigest(),
              main_commit='06b7755', identical_main_helpers=helpers,
              unchanged_native_extensions=True, comparison_pairs=0,
              comparison_reference_checks=0)
for prefix, count in (('integrated', 12),):
    rows = json.loads((ROOT / f'{prefix}-pairs.json').read_text())
    assert len(rows) == count
    for row in rows:
        assert len(row['raw']) == 80
        raw = row['raw']
        assert abs(statistics.median(b - a for a, b, _ in raw) - row['paired_delta_us']) < 1e-10
        empty = statistics.median(r[2] for r in raw)
        for col, key in ((0, 'previous_us'), (1, 'merged_us')):
            assert abs(statistics.median(r[col] for r in raw) - empty - row[key]) < 1e-10
        report['comparison_pairs'] += len(raw)
    logfile = 'compare.log' if prefix == 'integrated' else 'main-compare.log'
    log = (ROOT / logfile).read_text()
    assert len(re.findall(r'^PASS ', log, re.M)) == count * 6
    assert 'Traceback' not in log
    report['comparison_reference_checks'] += count * 6
    report[prefix + '_comparisons'] = [{k: v for k, v in row.items() if k != 'raw'} for row in rows]

suite = (ROOT / 'suite.log').read_text()
assert len(re.findall(r'^PASS \d+ ', suite, re.M)) == 16
assert len(re.findall(r'^(?:decode|verify) (?:1|16|32)\s+(?:full|sliding)\s+', suite, re.M)) == 12
assert 'Traceback' not in suite
report['suite_passes'] = 16
report['local_unpaired_bench'] = float(re.search(r'geomean over loads: ([\d.]+)', suite)[1])
fresh = json.loads((ROOT / 'integrated-fresh.json').read_text())
assert len(fresh) == 12 and all(r['replays_per_source'] == 4 for r in fresh)
fresh_log = (ROOT / 'fresh.log').read_text()
assert len(re.findall(r'^PASS ', fresh_log, re.M)) == 96 and 'Traceback' not in fresh_log
report['fresh_replays_per_source'] = 48
report['fresh_replays_total'] = 96
assert report['comparison_pairs'] == 960
assert report['comparison_reference_checks'] == 72
(ROOT / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))

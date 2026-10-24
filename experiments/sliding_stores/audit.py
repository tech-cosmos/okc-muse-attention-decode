"""Check archived source, correctness logs, timing pairs and machine-code evidence."""
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
R = ROOT / 'results'
production = Path('submission.py').read_text()
for n in ast.parse(production).body:
    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_NATIVE_SLIDING' for t in n.targets):
        cuda = next(ast.literal_eval(k.value) for k in n.value.keywords if k.arg == 'cuda_sources')
assert cuda == (ROOT / 'base.cu').read_text()
before = {n: (ROOT / (n + '.cu')).read_bytes() for n in ('vector', 'explicit')}
subprocess.run(['python', str(ROOT / 'generate.py')], check=True)
assert all((ROOT / (n + '.cu')).read_bytes() == s for n, s in before.items())
report = {'production_sha256': hashlib.sha256(production.encode()).hexdigest(), 'variants': {}}
for n in ('base', 'vector', 'explicit'):
    sass = (R / (n + '.sass')).read_text().split('Function : _Z7slidingILi1ELb1EE')[1]
    inst = re.findall(r'/\*[0-9a-f]+\*/\s+([^/]+?);', sass)
    report['variants'][n] = {'static_instructions': len(inst),
        'scalar_partial_stores': sum('STG.E ' in x for x in inst),
        'vector64_partial_stores': sum('STG.E.64 ' in x for x in inst),
        'qgmma': sum('QGMMA' in x for x in inst), 'hgmma': sum('HGMMA' in x for x in inst),
        'bar_sync': sum('BAR.SYNC' in x for x in inst)}
    assert 'REG:74 STACK:0 SHARED:17408 LOCAL:0' in (R / ('resources-' + n + '.txt')).read_text()
for n in ('vector', 'explicit'):
    test = (R / ('test-' + n + '.txt')).read_text()
    assert len(re.findall(r'^PASS \d+ ', test, re.M)) == 16
    assert not re.search(r'^FAIL|Traceback', test, re.M)
    bench = (R / ('bench-' + n + '.txt')).read_text()
    assert len(re.findall(r'^\w+ +\d+ +(?:sliding|full) ', bench, re.M)) == 12
    fresh = (R / ('fresh-' + n + '.txt')).read_text()
    assert fresh.count('PASS fresh graph') == 10 and 'Traceback' not in fresh
    pairs = json.loads((R / ('native-' + n + '.json')).read_text())
    assert len(pairs) == 10 and all(len(p['raw']) == 60 for p in pairs)
    report['variants'][n].update(tests=16, fresh_graph_replays=60,
        bench=bench.split('geomean over loads: ')[1].strip(),
        pairs=[{k: v for k, v in p.items() if k != 'raw'} for p in pairs])
for tool in ('memcheck', 'synccheck', 'racecheck'):
    s = (R / (tool + '-explicit.txt')).read_text()
    assert s.count('PASS native-only') == 4
    assert 'ERROR SUMMARY: 0 errors' in s or (tool == 'racecheck' and 'RACECHECK SUMMARY: 0 hazards' in s)
report['sanitizers'] = {'explicit': '12 native calls per tool, independent output/cache reference checks'}
for suffix, count in [('-confirm', 10), ('-fixed', 3)]:
    pairs = json.loads((R / ('native-explicit' + suffix + '.json')).read_text())
    assert len(pairs) == count and all(len(p['raw']) == 120 for p in pairs)
report['additional_comparisons'] = {'explicit_second_seed_shapes': 10,
    'explicit_fixed_length_shapes': 3, 'rounds_each': 120}
report['profiles'] = json.loads((R / 'profile-summary.json').read_text())
(R / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))

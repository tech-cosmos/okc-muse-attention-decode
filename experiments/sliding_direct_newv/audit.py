"""Reproduce sources and audit the session's recorded evidence."""
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
results = root / 'results'
source = Path('submission.py').read_text()
node = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == '_NATIVE_SLIDING' for t in n.targets))
native = next(ast.literal_eval(k.value) for k in node.value.keywords if k.arg == 'cuda_sources')
assert native == (root / 'base.cu').read_text()
paths = [root / f'{v}.{ext}' for v in ('direct', 'conditional') for ext in ('cu', 'py')]
before = [p.read_bytes() for p in paths]
subprocess.run([sys.executable, str(root / 'generate.py')], check=True)
assert before == [p.read_bytes() for p in paths]
report = {'production_sha256': hashlib.sha256(source.encode()).hexdigest(), 'variants': {}}
for name in ('base', 'direct', 'conditional'):
    sass = (results / f'{name}.sass').read_text().split('Function : _Z7slidingILi1ELb1EE')[1].split('Function :')[0]
    rec = {key: len(re.findall(pattern, sass)) for key, pattern in {
        'qgmma_sites': 'QGMMA', 'hgmma_sites': 'HGMMA', 'cta_barrier_sites': 'BAR.SYNC',
        'f2fp_sites': 'F2FP', 'instruction_sites_including_nops': r'/\*[0-9a-f]+\*/',
    }.items()}
    resource = (results / f'{name}.resources.txt').read_text()
    rec['resource_records'] = [line.strip() for line in resource.splitlines() if 'REG:' in line]
    assert len(rec['resource_records']) == 4
    assert all('STACK:0 SHARED:17408 LOCAL:0' in line for line in rec['resource_records'])
    assert rec['qgmma_sites'] == 2 and rec['hgmma_sites'] == 4
    assert rec['cta_barrier_sites'] == (2 if name == 'direct' else 3)
    if name != 'base':
        validation = (results / f'validate-{name}.txt').read_text()
        assert len(re.findall('^PASS [0-9]+ ', validation, re.M)) == 16
        assert 'FAIL' not in validation and 'Traceback' not in validation
        rec['tests'] = 16
        rec['broad_native_bench'] = re.search('geomean over loads: (.*)', validation).group(1)
        fresh = (results / f'fresh-{name}.txt').read_text()
        assert fresh.count('PASS fresh graph') == 10
        assert 'Traceback' not in fresh
        rec['fresh_native_graph_replays'] = 60
        pairs = json.loads((results / f'native-{name}.json').read_text())
        assert len(pairs) == 8 and all(len(x['raw']) == 60 for x in pairs)
        rec['pairs'] = [{k: v for k, v in x.items() if k != 'raw'} for x in pairs]
        for tool in ('memcheck', 'synccheck', 'racecheck'):
            text = (results / f'{tool}-{name}.txt').read_text()
            assert text.count('PASS native-only') == 4 and 'Traceback' not in text
            expected = ('RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)'
                        if tool == 'racecheck' else 'ERROR SUMMARY: 0 errors')
            assert expected in text, (name, tool)
        rec['sanitizers'] = '12 calls/tool: memcheck/synccheck zero errors; racecheck zero hazards/errors/warnings'
    report['variants'][name] = rec
assert (results / 'bitexact.txt').read_text().count('PASS bit-exact partials/cache') == 12
report['bit_exact_comparisons'] = 24
(results / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))

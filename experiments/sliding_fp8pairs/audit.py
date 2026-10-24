import hashlib
import json
import re
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parent
results = root / 'results'
source = Path('submission.py').read_text()
native = source.split('_NATIVE_SLIDING = load_inline(')[1].split("cuda_sources=r'''")[1].split("''',")[0]
assert native == (root / 'base.cu').read_text()
paths = [root / f'{v}.{ext}' for v in ('query', 'both') for ext in ('cu', 'py')]
before = [p.read_bytes() for p in paths]
subprocess.run(['python', str(root / 'generate.py')], check=True)
assert before == [p.read_bytes() for p in paths]
report = {'production_sha256': hashlib.sha256(source.encode()).hexdigest(), 'variants': {}}
for name in ('base', 'query', 'both'):
    sass = (results / f'{name}.sass').read_text().split('Function : _Z7slidingILi1ELb1EE')[1].split('Function :')[0]
    rec = {key: len(re.findall(pattern, sass)) for key, pattern in {
        'fp32_to_fp8_sites': 'F2FP.SATFINITE.E4M3.F32', 'all_f2fp_sites': 'F2FP',
        'prmt_sites': 'PRMT', 'lop3_sites': 'LOP3', 'qgmma_sites': 'QGMMA',
        'hgmma_sites': 'HGMMA', 'cta_barrier_sites': 'BAR.SYNC',
    }.items()}
    resources = (results / f'{name}.resources.txt').read_text()
    assert resources.count('REG:74 STACK:0 SHARED:17408 LOCAL:0') == 4
    rec.update(registers=74, shared_bytes_cuobjdump=17408, stack_bytes=0, local_bytes=0)
    if name != 'base':
        text = (results / f'validate-{name}.txt').read_text()
        assert len(re.findall('^PASS [0-9]+ ', text, re.M)) == 16
        assert 'FAIL' not in text and 'Traceback' not in text
        rec['tests'] = 16
        rec['broad_native_bench'] = re.search('geomean over loads: (.*)', text).group(1)
        fresh = (results / f'fresh-{name}.txt').read_text()
        assert fresh.count('PASS fresh graph') == 10
        rec['fresh_native_graph_replays'] = 60
        pairs = json.loads((results / f'native-{name}.json').read_text())
        assert len(pairs) == 8 and all(len(x['raw']) == 60 for x in pairs)
        rec['pairs'] = [{k: v for k, v in x.items() if k != 'raw'} for x in pairs]
    report['variants'][name] = rec
for tool in ('memcheck', 'synccheck', 'racecheck'):
    text = (results / f'{tool}-both.txt').read_text()
    assert text.count('PASS native-only') == 4
    expected = 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' if tool == 'racecheck' else 'ERROR SUMMARY: 0 errors'
    assert expected in text, tool
report['both_sanitizers'] = 'memcheck/synccheck zero errors; racecheck zero hazards/errors/warnings; 12 native calls each'
assert (results / 'bitexact.txt').read_text().count('PASS bit-exact partials/cache') == 12
report['bit_exact_comparisons'] = 24
report['source_metrics'] = json.loads((results / 'profile-summary.json').read_text())
assert Path('profiles/session21-integrated-test.txt').read_text().count('PASS ') == 16
assert Path('profiles/session21-integrated-fresh.txt').read_text().count('PASS fresh graph') == 8
integrated = json.loads((results / 'native-integrated.json').read_text())
assert len(integrated) == 5 and all(len(x['raw']) == 60 for x in integrated)
report['integration'] = {'tests': 16, 'fresh_graph_replays': 24,
    'pairs': [{k:v for k,v in x.items() if k != 'raw'} for x in integrated]}
(results / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))

"""CPU audit of exact sources and the archived validation evidence."""
import hashlib
import csv
import importlib.util
import json
from pathlib import Path
import re
import statistics
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
spec = importlib.util.spec_from_file_location('generator', ROOT / 'generate.py')
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)
assert (ROOT / 'deferred.cu').read_text() == gen.generate()
assert (ROOT / 'lane.cu').read_text() == gen.generate(True)
assert (ROOT / 'natural_v.cu').read_bytes() == subprocess.check_output([
    'git', 'show', 'fded6bf:experiments/transposed_decode/natural_v.cu'])
frozen = (ROOT / 'production.py').read_bytes()
assert frozen == subprocess.check_output(['git', 'show', '9dbf51a:submission.py'])

report = dict(production_sha256=hashlib.sha256(frozen).hexdigest(),
              source_reproduction=True, pairs=0, comparison_reference_checks=0,
              fresh_native_replays=0, fresh_production_replays=0, variants={})
for variant in ('deferred', 'lane'):
    suite = (ROOT / (variant + '-suite.log')).read_text()
    assert len(re.findall(r'^PASS \d+ ', suite, re.M)) == 16
    assert len(re.findall(r'^(?:decode|verify) (?:1|16|32)\s+(?:full|sliding)\s+', suite, re.M)) == 12
    assert 'Traceback' not in suite
    result = dict(suite_passes=16, suite_native_cases=5, suite_fallback_cases=11,
                  local_bench=float(re.search(r'geomean over loads: ([\d.]+)', suite)[1]), comparisons={})
    for control in ('production', 'natural'):
        rows = json.loads((ROOT / f'{variant}-{control}-pairs.json').read_text())
        assert len(rows) == 8
        log = (ROOT / f'{variant}-{control}-compare.log').read_text()
        assert len(re.findall(r'^PASS ', log, re.M)) == 48
        assert 'Traceback' not in log
        report['comparison_reference_checks'] += 48
        for row in rows:
            assert len(row['raw']) == 60
            delta = statistics.median(b - a for a, b, _ in row['raw'])
            assert abs(delta - row['paired_delta_us']) < 1e-10
            empty = statistics.median(r[2] for r in row['raw'])
            for idx, key in ((0, 'previous_us'), (1, 'merged_us')):
                assert abs(statistics.median(r[idx] for r in row['raw']) - empty - row[key]) < 1e-10
            report['pairs'] += len(row['raw'])
        result['comparisons'][control] = [{k: v for k, v in row.items() if k != 'raw'} for row in rows]
    fresh = json.loads((ROOT / f'{variant}-fresh.json').read_text())
    assert len(fresh) == 8 and all(r['replays_per_source'] == 4 for r in fresh)
    log = (ROOT / f'{variant}-fresh.log').read_text()
    assert len(re.findall(r'^PASS ', log, re.M)) == 64 and 'Traceback' not in log
    report['fresh_native_replays'] += 32
    report['fresh_production_replays'] += 32
    for tool in ('memcheck', 'synccheck', 'racecheck'):
        log = (ROOT / f'{variant}-{tool}.log').read_text()
        assert 'TOTAL 24' in log and len(re.findall(r'^PASS ', log, re.M)) == 24
        expected = ('RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)'
                    if tool == 'racecheck' else 'ERROR SUMMARY: 0 errors')
        assert expected in log and 'Traceback' not in log
    result['sanitizers'] = dict(replays_per_tool=24, memcheck_errors=0,
                                synccheck_errors=0, racecheck_hazards=0)
    report['variants'][variant] = result

partials = json.loads((ROOT / 'partials.json').read_text())
assert len(partials) == 32
assert all(r['exact_numerator'] and r['exact_maxima'] and r['exact_cache'] for r in partials)
report['partial_comparisons'] = len(partials)
report['max_denominator_relative_error'] = max(r['max_relative_denominator_error'] for r in partials)
assert report['max_denominator_relative_error'] < 1e-5
report['resources'] = {}
for variant in ('natural', 'deferred', 'lane'):
    resources = (ROOT / f'{variant}-resources.txt').read_text()
    assert resources.count('STACK:0 SHARED:1024 LOCAL:0') == 2
    regs = re.findall(r'Function .*?Li(1|64)EE.*?\n\s+REG:(\d+)', resources)
    assert len(regs) == 2
    sass = (ROOT / f'{variant}-sass.txt').read_text()
    page1 = next(block for block in sass.split('Function :') if block.startswith(' _Z6decodeILi1EE'))
    ops = re.findall(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)', page1)
    assert len(ops) > 100
    ncu = (ROOT / f'{variant}-ncu.log').read_text()
    assert f'PASS profile {variant}' in ncu and 'Traceback' not in ncu
    metrics = {r[-3]: float(r[-1].replace(',', ''))
               for r in csv.reader(line for line in ncu.splitlines() if line.startswith('"0"'))}
    assert len(metrics) == 7
    report['resources'][variant] = dict(registers_by_page=dict(regs), static_shared_bytes=1024,
                                       dynamic_shared_bytes=69632, stack_bytes=0, local_bytes=0,
                                       page1_static_instruction_sites=len(ops),
                                       page1_qgmma=sum(x.startswith('QGMMA') for x in ops),
                                       page1_hgmma=sum(x.startswith('HGMMA') for x in ops),
                                       page1_shfl=sum(x.startswith('SHFL') for x in ops),
                                       ncu_metrics=metrics)
assert report['pairs'] == 1920
assert report['comparison_reference_checks'] == 192
assert report['fresh_native_replays'] == 64
assert report['fresh_production_replays'] == 64
(ROOT / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))

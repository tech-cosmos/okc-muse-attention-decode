"""Summarize source counters; keep them distinct from hardware counters."""
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def number(value):
    return float(value.replace(',', ''))


def source_rows(path):
    text = path.read_text()
    return list(csv.DictReader(io.StringIO(text[text.index('"Address",'):])) )


def summarize(source, hardware):
    rows = source_rows(source)
    cp = [r for r in rows if 'LDGSTS' in r['Source']]
    excess = lambda rs: sum(int(number(r['L1 Wavefronts Shared Excessive'])) for r in rs)
    raw = list(csv.DictReader(hardware.open()))[1]
    return {
        'copy_instruction_sites': len(cp),
        'copy_source_excess': excess(cp),
        'all_source_excess': excess(rows),
        'copy_source_ideal': sum(int(number(r['L1 Wavefronts Shared Ideal'])) for r in cp),
        'hardware_conflicts': {
            op: int(number(raw[f'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_{op}.sum']))
            for op in ('ldgsts', 'ld', 'st', 'gmma')
        },
    }


production = (ROOT.parents[1] / 'submission.py').read_text()
cuda = production.split('_NATIVE_SLIDING = load_inline(', 1)[1].split("cuda_sources=r'''", 1)[1].split("'''", 1)[0]
assert cuda == (ROOT / 'base.cu').read_text()
report = {'production_sha256': hashlib.sha256(production.encode()).hexdigest(),
          'native_source_equals_production': True, 'probes': {}}
for src in range(3):
    for dst in range(3):
        stem = ROOT / 'results' / f'copy-{src}-{dst}'
        text = stem.with_suffix('.csv').read_text()
        assert f'PASS source={src} dest={dst} fresh_graphs=3 bytes_per_graph=262144' in text
        report['probes'][f'{src}-{dst}'] = summarize(stem.with_suffix('.csv'), stem.with_name(stem.name+'-hardware.csv'))
report['native'] = summarize(ROOT/'results/base-source.csv', ROOT/'results/base-hardware.csv')
report['fresh_graphs_per_probe'] = 3
report['fresh_graphs_total'] = 27
report['diagnostic_sanitizers'] = {}
for tool in ('memcheck', 'synccheck', 'racecheck'):
    log = (ROOT/'results'/f'{tool}.txt').read_text()
    assert 'PASS source=2 dest=1 fresh_graphs=3 bytes_per_graph=262144' in log
    expected = 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' if tool == 'racecheck' else 'ERROR SUMMARY: 0 errors'
    assert expected in log
    report['diagnostic_sanitizers'][tool] = 'permuted source / SW64 only: 3 fresh graphs, 0 errors/hazards'
assert report['probes']['0-0']['copy_source_excess'] == 0
assert report['probes']['0-1']['copy_source_excess'] == 0
assert report['probes']['1-0']['copy_source_excess'] == 2048
assert report['probes']['1-1']['copy_source_excess'] == 2048
assert report['probes']['2-0']['copy_source_excess'] == 2048
assert report['probes']['2-1']['copy_source_excess'] == 2048
assert all(report['probes'][f'{s}-2']['copy_source_excess'] == 14336 for s in range(3))
assert report['native']['copy_source_excess'] == report['native']['all_source_excess'] == 4104
(ROOT/'results/audit.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))

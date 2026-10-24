"""CPU-only reproducibility, source-scope, assembly and result audit."""
import ast
import hashlib
import json
import re
import statistics
from pathlib import Path

from generate import generate

ROOT = Path(__file__).resolve().parent
previous = (ROOT/'previous.py').read_text()
candidate = (ROOT/'candidate.py').read_text()
assert candidate == generate(previous)

def nodes(source):
    return [ast.dump(n, include_attributes=False) for n in ast.parse(source).body
            if not (isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == '_NATIVE_MERGE' for t in n.targets))]

assert nodes(previous) == nodes(candidate)
result = {'production_sha256': hashlib.sha256(previous.encode()).hexdigest(),
          'candidate_sha256': hashlib.sha256(candidate.encode()).hexdigest(),
          'unchanged_top_level_nodes': len(nodes(previous)), 'assembly': {}}
for name in ('previous', 'candidate'):
    functions = []
    for part in (ROOT/(name+'.sass')).read_text().split('Function : ')[1:]:
        lines = [line.strip() for line in part.splitlines()
                 if re.match(r'\s*/\*[0-9a-f]+\*/', line)]
        loads = [i for i, line in enumerate(lines) if 'LDG.E.64.STRONG.GPU' in line]
        wait = [i for i, line in enumerate(lines) if 'ACQBULK' in line]
        first_max = next(i for i, line in enumerate(lines) if 'FMNMX' in line)
        first_shuffle = next(i for i, line in enumerate(lines) if 'SHFL.BFLY' in line)
        first_exp = next(i for i, line in enumerate(lines) if 'MUFU.EX2' in line)
        assert len(loads) == 17
        assert not wait or (len(wait) == 1 and min(loads) > wait[0])
        functions.append(dict(function=part.splitlines()[0], instruction_sites=len(lines),
                              wait=wait, coherent_loads=loads, first_max=first_max,
                              first_shuffle=first_shuffle, first_exp=first_exp,
                              loads_before_first_max=sum(i < first_max for i in loads),
                              loads_before_first_exp=sum(i < first_exp for i in loads)))
    assert len(functions) == 2
    resource = (ROOT/(name+'-resources.txt')).read_text()
    assert resource.count('REG:48 STACK:0 SHARED:3200 LOCAL:0') == 2
    result['assembly'][name] = functions

suite = (ROOT/'suite.log').read_text()
assert len(re.findall(r'^PASS \d', suite, re.M)) == 16
assert len(re.findall(r'^(?:decode|verify)\s+\d+\s+(?:full|sliding)', suite, re.M)) == 12
result['suite_passes'] = 16
result['bench_shapes'] = 12
result['comparisons'] = {}
total_pairs = 0
total_checks = 0
for mode in ('compare', 'confirm'):
    path = ROOT/('pairs-'+mode+'.json')
    if not path.exists():
        continue
    rows = json.loads(path.read_text())
    checks = len(re.findall(r'^PASS ', (ROOT/(mode+'.log')).read_text(), re.M))
    assert checks == 6*len(rows)
    for row in rows:
        raw = row['raw']
        assert len(raw) == 80
        assert row['paired_delta_us'] == statistics.median(b-a for a, b, _ in raw)
        cost = statistics.median(e for _, _, e in raw)
        assert row['previous_us'] == statistics.median(a for a, _, _ in raw)-cost
        assert row['merged_us'] == statistics.median(b for _, b, _ in raw)-cost
    total_pairs += len(rows)*80
    total_checks += checks
    result['comparisons'][mode] = [{k: v for k, v in row.items() if k != 'raw'} for row in rows]
fresh = json.loads((ROOT/'fresh.json').read_text())
fresh_count = sum(row['replays_per_source']*2 for row in fresh)
assert fresh_count == len(re.findall(r'^PASS ', (ROOT/'fresh.log').read_text(), re.M))
result.update(adjacent_pairs=total_pairs, reference_checks_in_pairs=total_checks,
              fresh_graph_replays=fresh_count)
(ROOT/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k: v for k, v in result.items() if k not in ('assembly', 'comparisons')}, indent=2))

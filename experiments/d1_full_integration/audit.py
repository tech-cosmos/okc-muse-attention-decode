"""CPU-only source-integration and archived-result checks."""
import ast
import hashlib
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sources = {name: (ROOT/(name+'.py')).read_text() for name in ('previous','upstream','integrated')}
trees = {name: ast.parse(source) for name, source in sources.items()}

def aststr(node):
    return ast.dump(node, include_attributes=False)

def definitions(tree):
    return {node.name: aststr(node) for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}

defs = {name: definitions(tree) for name, tree in trees.items()}
shared = set(defs['upstream'])-{'kernel_fn'}
for name in shared:
    assert defs['upstream'][name] == defs['integrated'][name], name
for name in set(defs['previous'])-{'_plan'}:
    assert defs['previous'][name] == defs['integrated'][name], name

def native_assignments(tree):
    return [aststr(node) for node in tree.body if isinstance(node,ast.Assign) and any(
        isinstance(t,ast.Name) and t.id in ('_NATIVE_SLIDING','_NATIVE_MERGE') for t in node.targets)]

assert native_assignments(trees['previous']) == native_assignments(trees['integrated'])
assert len(native_assignments(trees['integrated'])) == 2
result = dict(sha256={name: hashlib.sha256(source.encode()).hexdigest() for name,source in sources.items()},
              shared_definitions=len(shared), native_definitions_unchanged=2, dispatcher_unchanged=True)
suite = (ROOT/'suite.log').read_text()
assert len(re.findall(r'^PASS \d', suite, re.M)) == 16
assert len(re.findall(r'^(?:decode|verify)\s+\d+\s+(?:full|sliding)',suite,re.M)) == 12
result.update(suite_passes=16, bench_shapes=12, comparisons={})
pairs = checks = 0
for mode in ('compare','confirm'):
    rows = json.loads((ROOT/('pairs-'+mode+'.json')).read_text())
    count = len(re.findall(r'^PASS ',(ROOT/(mode+'.log')).read_text(),re.M))
    assert count == len(rows)*9
    checks += count
    summary=[]
    for row in rows:
        contrasts={}
        for name, contrast in row['contrasts'].items():
            raw=contrast['raw'];assert len(raw)==80
            cost=statistics.median(e for _,_,e in raw)
            assert contrast['parent_us']==statistics.median(a for a,_,_ in raw)-cost
            assert contrast['integrated_us']==statistics.median(b for _,b,_ in raw)-cost
            assert contrast['paired_delta_us']==statistics.median(b-a for a,b,_ in raw)
            pairs+=len(raw)
            contrasts[name]={k:v for k,v in contrast.items() if k!='raw'}
        summary.append(dict(shape=row['shape'],dispatch=row['dispatch'],contrasts=contrasts))
    result['comparisons'][mode]=summary
fresh=json.loads((ROOT/'fresh.json').read_text())
fresh_count=sum(len(row['sources'])*row['replays_per_source'] for row in fresh)
assert fresh_count==len(re.findall(r'^PASS ',(ROOT/'fresh.log').read_text(),re.M))
result.update(adjacent_pairs=pairs, reference_checks_in_pairs=checks, fresh_graph_replays=fresh_count)
(ROOT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='comparisons'},indent=2))

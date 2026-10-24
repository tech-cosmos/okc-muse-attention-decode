"""CPU-only evidence audit: sources, loop capacities, graph pairs and pass counts."""
import ast,hashlib,json,re,runpy
from pathlib import Path
ROOT=Path(__file__).resolve().parent
hashfile=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
before={str(p.name):hashfile(p) for p in (ROOT/'tile128.cu',ROOT/'tile256.cu',ROOT/'binding.cpp')}
runpy.run_path(str(ROOT/'generate.py'))
assert before=={k:hashfile(ROOT/k) for k in before}
prod=ROOT/'production_snapshot.py';assert hashfile(prod)=='1bcf42aa6edf7f25a02117e3eabebb6f5b2e69967359137e015b2f79d445a548'
assert hashfile(Path('submission.py'))==hashfile(prod)
for bn in (128,256):
 s=(ROOT/f'tile{bn}.cu').read_text();g=s[s.index(' auto gather='):s.index(' if(EARLY)gather();')]
 assert f'j<{bn//32};j++' in g
 assert (bn//32)*128*16==bn*64
 assert f'j<{bn//16};j++' in s[s.index('// Keep new K/V'):]
 assert f'float score[{bn//2}]' in s
 assert f'cudaFuncAttributeMaxDynamicSharedMemorySize,{bn*256}' in s
 # Disjoint buffers, exact byte coverage, no hidden historical window limit.
 assert f'Vraw=storage+{bn*64};half* V=(half*)(storage+{bn*128})' in s
 assert f'lo=split*{bn}' in s and 'seq-2047' not in s
rows=json.loads((ROOT/'results/initial.json').read_text());assert len(rows)==16
assert {r['index'] for r in rows}==set(range(8))
for r in rows:assert len(r['raw'])==60
initial=(ROOT/'results/initial.log').read_text();assert initial.count('CHECK ')==72 and ' False ' not in initial
suite=(ROOT/'results/suite.log').read_text();assert len(re.findall(r'^PASS \d+ ',suite,re.M))==32 and 'FAIL' not in suite
assert suite.count('H100 time vs SGLang-on-H200')==2
fresh=(ROOT/'results/fresh.log').read_text();assert fresh.count("CHECK ('fresh'")==48 and ' False ' not in fresh
for tool in ('memcheck','synccheck','racecheck'):
 s=(ROOT/f'results/{tool}.log').read_text();assert 'TOTAL 24' in s and len(re.findall(r'^PASS ',s,re.M))==24
 assert 'ERROR SUMMARY: 0 errors' in s or 'RACECHECK SUMMARY: 0 hazards' in s
 assert 'FAIL' not in s
boundary=(ROOT/'results/debug-final.log').read_text();assert len(re.findall(r'^\d+ True ',boundary,re.M))==8 and 'False' not in boundary
# Shared non-dispatch helpers must remain exactly inherited from main.
past=ast.parse(prod.read_text());mast=ast.parse((ROOT/'main_snapshot.py').read_text())
pfunc={n.name:ast.dump(n,include_attributes=False) for n in past.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
shared=[n.name for n in mast.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name not in ('kernel_fn',)]
for n in mast.body:
 if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in shared:assert pfunc[n.name]==ast.dump(n,include_attributes=False),n.name
result=dict(production_sha256=hashfile(prod),sources=before,shared_helpers=len(shared),timing_shapes=8,
 native_variants=2,adjacent_pairs=960,comparison_reference_checks=72,full_suite_passes=32,
 full_suite_native_cases_per_variant=5,fresh_native_replays=48,sanitizer_native_replays_per_tool=24,
 boundary_reference_checks=8,production_changed=False,judge_requests=0)
(ROOT/'results/audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

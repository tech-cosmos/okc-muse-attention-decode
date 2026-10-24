"""CPU source, SASS-ordering, and raw-results audit for session37."""
import ast, hashlib, json, re, statistics, runpy, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def parse(name):return ast.parse((ROOT/name).read_text())
def functions(tree):return {n.name:ast.dump(n) for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
def assignment(tree,name):return next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
old,new,main=map(parse,('previous.py','candidate.py','main.py'))
fo,fn,fm=map(functions,(old,new,main))
shared=set(fn)&set(fm)-{'kernel_fn'}
assert len(shared)==20
assert all(fn[k]==fm[k] for k in shared)
assert all(fn[k]==fo[k] for k in fn if k!='kernel_fn')
assert ast.dump(assignment(old,'_NATIVE_SLIDING'))==ast.dump(assignment(new,'_NATIVE_SLIDING'))
initial=(ROOT/'initial.py').read_text()
expected=initial.replace('muse_native_merge_bf16','muse_native_merge_bf16_bits').replace('return make_float2(__bfloat162float(__ushort_as_bfloat16(v & 65535)),\n                   __bfloat162float(__ushort_as_bfloat16(v >> 16)));','return make_float2(__uint_as_float(v << 16), __uint_as_float(v & 0xffff0000u));')
assert expected==(ROOT/'bits.py').read_text()
assert runpy.run_path(str(ROOT/'generate_packed.py'))['s']==(ROOT/'candidate.py').read_text()
assert (ROOT/'previous.py').read_bytes()==subprocess.check_output(['git','show','d125df9:submission.py'])
assert (ROOT/'main.py').read_bytes()==subprocess.check_output(['git','show','06b7755:submission.py'])
assert functions(parse('initial.py'))==functions(parse('bits.py'))==functions(new)
# Exhaustive BF16 bit widening, including signed zero, subnormals, inf/NaN payloads.
for packed in range(65536):
    assert ((packed|((65535-packed)<<16))<<16)&0xffffffff==packed<<16
    assert (packed|((65535-packed)<<16))&0xffff0000==(65535-packed)<<16
sass_results={}
for f in ('native-initial.sass','native-bits.sass','native.sass'):
    results=[]
    for part in (ROOT/f).read_text().split('Function : ')[1:]:
        name=part.splitlines()[0]
        inst=[x for x in part.splitlines() if re.search(r'/\*[0-9a-f]+\*/',x)]
        loads=[i for i,x in enumerate(inst) if 'LDG.' in x and '.STRONG.GPU' in x]
        assert len(loads)==17,(name,len(loads))
        waits=[i for i,x in enumerate(inst) if 'ACQBULK' in x]
        if 'ELb1E' in name:assert len(waits)==1 and waits[0]<min(loads)
        else:assert not waits
        results.append(dict(name=name,instructions=len(inst),coherent_loads=len(loads),wait_indices=waits))
    assert len(results)==4
    sass_results[f]=results
resources={}
for f in ('resources-initial.txt','resources-bits.txt','resources.txt'):
    text=(ROOT/f).read_text()
    rs=[tuple(map(int,x)) for x in re.findall(r'REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)',text)]
    assert len(rs)==4 and all(stack==local==0 for _,stack,_,local in rs)
    resources[f]=rs
results={}
for f in sorted(ROOT.glob('pairs-*.json')):
    data=json.loads(f.read_text())
    for row in data:
        raw=row['raw'];assert len(raw)==80
        cost=statistics.median(x[2] for x in raw)
        assert abs(row['previous_us']-(statistics.median(x[0] for x in raw)-cost))<1e-9
        assert abs(row['merged_us']-(statistics.median(x[1] for x in raw)-cost))<1e-9
        assert abs(row['paired_delta_us']-statistics.median(x[1]-x[0] for x in raw))<1e-9
    results[f.name]=dict(cases=len(data),pairs=sum(len(r['raw']) for r in data),reference_checks=len(data)*6)
def bodies(file,original=False):
    return {('ELb1E' in part.splitlines()[0]):[x.split('*/',1)[1].split(';',1)[0].strip() for x in part.splitlines() if re.search(r'/\*[0-9a-f]+\*/',x)]
            for part in (ROOT/file).read_text().split('Function : ')[1:] if original or re.search(r'ELb[01]EfE',part.splitlines()[0])}
assert bodies('previous.sass',True)==bodies('native.sass')
for filename,logname,count in [('pairs-initial.json','compare-initial.log',19),('pairs-bits.json','compare-bits.log',23),('pairs-37571.json','compare.log',23),('pairs-37572.json','confirm.log',6)]:
    data=json.loads((ROOT/filename).read_text());assert len(data)==count
    log=(ROOT/logname).read_text();assert len(re.findall(r'^PASS ',log,re.M))==6*count and 'Traceback' not in log
for f in ('suite-initial.log','suite-bits.log','suite.log'):
    log=(ROOT/f).read_text()
    assert len(re.findall(r'^PASS \d+ ',log,re.M))==16 and 'Traceback' not in log
    assert len(re.findall(r'^(?:decode|verify) (?:1|16|32)\s+(?:full|sliding)\s+',log,re.M))==12
fresh=json.loads((ROOT/'fresh.json').read_text());assert len(fresh)==23
assert all(r['replays_per_source']==4 for r in fresh)
assert len(re.findall(r'^PASS ',(ROOT/'fresh.log').read_text(),re.M))==184
native_fresh=sum(bool(r['native_calls']['merged'])*r['replays_per_source'] for r in fresh)
synthetic={}
for label in ('plain','memcheck','synccheck','racecheck'):
    f=ROOT/('synthetic-'+label+'.json')
    if not f.exists():continue
    data=json.loads(f.read_text());assert len(data)==24
    assert sum(r['replays'] for r in data)==72
    assert all(max(r['relative_l2'])<0.003 for r in data)
    log=(ROOT/('synthetic.log' if label=='plain' else label+'.log')).read_text()
    assert 'PASS 24 cases 72 fresh graph replays' in log and 'Traceback' not in log
    if label!='plain':
        assert ('ERROR SUMMARY: 0 errors' in log or 'RACECHECK SUMMARY: 0 hazards' in log)
    synthetic[label]=dict(cases=24,replays=72,max_rel_l2=max(max(r['relative_l2']) for r in data))
result=dict(source_sha256=hashlib.sha256((ROOT/'candidate.py').read_bytes()).hexdigest(),shared_helpers=len(shared),native_sliding_unchanged=True,
            conversion_exact=True,float_sass_identical_to_previous=True,sass=sass_results,resources=resources,pair_results=results,
            paired_graph_pairs=sum(r['pairs'] for r in results.values()),comparison_reference_checks=sum(r['reference_checks'] for r in results.values()),
            fresh_replays=184,fresh_native_candidate_replays=native_fresh,synthetic=synthetic)
(ROOT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

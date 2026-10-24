"""CPU-only source, result, and machine-code audit; no claim beyond recorded runs."""
import ast,csv,hashlib,io,json,re,statistics,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((ROOT/'results/source-manifest.json').read_text())
assert all(sha(ROOT/p)==value for p,value in manifest.items())
subprocess.run([sys.executable,str(ROOT/'generate.py')],check=True)
assert json.loads((ROOT/'results/source-manifest.json').read_text())==manifest
assert sha(Path('submission.py'))==manifest['production_snapshot.py']=='1bcf42aa6edf7f25a02117e3eabebb6f5b2e69967359137e015b2f79d445a548'
parent=ast.parse((ROOT/'production_snapshot.py').read_text())
def funcs(tree):
    return {n.name:ast.dump(n) for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)}
summaries={}
for variant,compare,fresh,suite in [('candidate','initial','fresh','suite'),('compensated','compensated-initial','compensated-fresh','compensated-suite')]:
    child=ast.parse((ROOT/(variant+'.py')).read_text());p,c=funcs(parent),funcs(child)
    same=[name for name in p if name!='kernel_fn']
    assert all(p[name]==c[name] for name in same)
    # The sole dispatcher change is the native config-only frequency helper.
    expected=(ROOT/'production_snapshot.py').read_text().replace('cs = _cos_sin(cfg["rope_theta"], 64, device)','cs = _rope_freq(cfg["rope_theta"], device)',1)
    assert funcs(ast.parse(expected))['kernel_fn']==c['kernel_fn']
    cuda=(ROOT/(variant+'.cu')).read_text()
    assert 'cs[(long)seq' not in cuda
    assert cuda.count('__syncthreads();')==(ROOT/'parent.cu').read_text().count('__syncthreads();')==3
    assert cuda.count('cp16(K+')==1 and cuda.count('cp16(Vraw+')==1
    assert 'for(int j=0;j<2;j++)' in cuda
    results=json.loads((ROOT/'results'/f'{compare}.json').read_text())
    assert len(results)==36
    rows=[]
    for r in results:
        assert len(r['raw'])==60
        assert abs(r['paired_delta_us']-statistics.median(b-a for a,b,e in r['raw']))<1e-12
        if r['variant']=='candidate':rows.append({k:v for k,v in r.items() if k!='raw'})
    freshrows=json.loads((ROOT/'results'/f'{fresh}.json').read_text())
    assert len(freshrows)==18 and sum(r['replays'] for r in freshrows)==72
    log=(ROOT/'results'/f'{suite}.log').read_text()
    assert len(re.findall(r'^PASS \d+ ',log,re.M))==16
    assert len(re.findall(r'^(decode|verify) ',log,re.M))==12
    accuracy=json.loads((ROOT/'results'/f'{variant}-accuracy.json').read_text())
    assert len(accuracy)==3 and all(r['angles']==2**21*32 and max(r['max_abs'])<1e-6 for r in accuracy)
    config=json.loads((ROOT/'results'/f'{variant}-config.json').read_text())
    assert len(config)==4 and all(r['eager']==1 and r['replays']==2 for r in config)
    for tool in ('memcheck','synccheck','racecheck'):
        s=(ROOT/'results'/f'{variant}-{tool}.log').read_text()
        assert s.count('PASS native-only')==24
        if tool=='racecheck':assert 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in s
        else:assert 'ERROR SUMMARY: 0 errors' in s
    sass=(ROOT/'results'/f'{variant}.sass.txt').read_text()
    parts=re.split(r'Function : ',sass)
    body=next(part for part in parts if part.startswith('_Z7slidingILi1ELb1'))
    instructions=re.findall(r'/\*[0-9a-f]+\*/\s+(.*?)\s*;\s*/\*',body)
    counts={op:sum(op in x for x in instructions) for op in ('MUFU.SIN','MUFU.COS','DFMA','DMUL','F2F.F64.F32','F2F.F32.F64','QGMMA','HGMMA','BAR.SYNC')}
    assert counts['MUFU.SIN']==counts['MUFU.COS']==10
    assert counts['QGMMA']==2 and counts['HGMMA']==4 and counts['BAR.SYNC']==3
    if variant=='compensated':assert counts['DFMA']==counts['DMUL']==counts['F2F.F64.F32']==counts['F2F.F32.F64']==0
    resources=(ROOT/'results'/f'{variant}.resources.txt').read_text()
    assert len(re.findall('REG:74 STACK:0 SHARED:17408 LOCAL:0',resources))==4
    summaries[variant]=dict(unchanged_helpers=len(same),suite_passes=16,native_suite_cases=3,fallback_cases=13,bench_shapes=12,
        comparison_pairs=2160,comparison_references=162,fresh_replays=72,alternate_config_checks=12,
        arithmetic_angles=sum(r['angles'] for r in accuracy),maximum_trig_error=max(max(r['max_abs']) for r in accuracy),
        isolated_native_sanitizer_replays_per_tool=24,static_instructions=len(instructions),static_operations=counts,
        timings=rows,control_delta_range=[min(r['paired_delta_us'] for r in results if r['variant']=='control'),max(r['paired_delta_us'] for r in results if r['variant']=='control')])
metrics={}
for name in ('parent','computed','compensated'):
    log=(ROOT/'results'/f'ncu-{name}.log').read_text()
    assert "PASS ('profile'," in log
    table=log[log.index('"ID","Process ID"'):]
    metrics[name]={r['Metric Name']:int(r['Metric Value'].replace(',','')) for r in csv.DictReader(io.StringIO(table))}
    assert len(metrics[name])==5 and metrics[name]['launch__registers_per_thread']==74
report=dict(production_sha256=manifest['production_snapshot.py'],sources=manifest,variants=summaries,ncu_metrics=metrics)
(ROOT/'results/audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='timings'} for k,v in summaries.items()},indent=2))

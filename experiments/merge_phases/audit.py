"""CPU-only provenance, assembly, trace indexing and result-count audit."""
import ast,gzip,hashlib,json,re,statistics,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(R/'previous.py')==sha(Path('submission.py'))=='ed5bde6e29882a90249d327c30602a9b07e83ad0d710288420c7a6c7eb23f9ca'
before=sha(R/'candidate.py');subprocess.run([sys.executable,str(R/'generate.py')],check=True);assert before==sha(R/'candidate.py')
old=ast.parse((R/'previous.py').read_text());new=ast.parse((R/'candidate.py').read_text())
for a,b in zip(old.body,new.body):
    if isinstance(a,ast.Assign) and isinstance(a.targets[0],ast.Name) and a.targets[0].id=='_NATIVE_MERGE':continue
    assert ast.dump(a)==ast.dump(b),getattr(a,'name',type(a).__name__)
subprocess.run([sys.executable,str(R/'analyze.py')],check=True,stdout=subprocess.DEVNULL)
asm=json.loads((R/'sass-analysis.json').read_text())
for row in asm:
    if row['phase']==2 and row['pdl']:assert row['between_clocks']==['ACQBULK;']
    if row['phase']==9:assert not row['between_clocks']
counts=dict(adjacent_graph_pairs=0,comparison_reference_checks=0,trace_replays=0)
for mode,ncase in [('compare',6),('confirm',4)]:
    rows=json.loads(gzip.decompress((R/f'pairs-{mode}.json.gz').read_bytes()));assert len(rows)==ncase*11
    assert (R/f'{mode}.log').read_text().count('PASS (')==ncase*11*6
    counts['comparison_reference_checks']+=ncase*11*6
    summaries=json.loads((R/f'summary-{mode}.json').read_text())
    assert summaries==[{k:v for k,v in x.items() if k not in ('raw','traces')} for x in rows]
    for row in rows:
        raw=row['raw'];assert len(raw)==60;counts['adjacent_graph_pairs']+=60
        med=statistics.median
        assert row['paired_delta_us']==med(x[1]-x[0] for x in raw)
        assert row['previous_us']==med(x[0] for x in raw)-med(x[2] for x in raw)
        assert row['probe_us']==med(x[1] for x in raw)-med(x[2] for x in raw)
        ts=row['traces'];p=row['phase'];assert len(ts)==(12 if p>0 else 0);counts['trace_replays']+=len(ts)
        shape=row['shape'];nr=shape['batch']*shape['q_len']*16
        offsets=[(r*8+w)*2+j for r in range(nr) for w in range(8) for j in range(2)]
        assert offsets==list(range(nr*16))
        for sample in ts:
            assert len(sample)==nr
            for r in sample:
                assert len(r)==8
                for w,(t0,t1) in enumerate(r):
                    if p in (6,7,8) and w:assert t1==0
                    else:assert t1>=t0>0
        if p>0:
            expected=[]
            for w in range(8):
                vals=[r[w][1]-r[w][0] for t in ts for r in t if r[w][1]!=0]
                expected.append(med(vals) if vals else None)
            assert row['warp_cycles']==expected
fresh=json.loads((R/'fresh.json').read_text());assert len(fresh)==6 and sum(x['replays'] for x in fresh)==240
assert (R/'fresh.log').read_text().count("PASS ('fresh'")==240
suite=(R/'suite.log').read_text();assert len(re.findall(r'^PASS \d+ ',suite,re.M))==16
assert len(re.findall(r'^(decode|verify) ',suite,re.M))==12
assert 'Traceback' not in suite and 'FAIL' not in suite
resources=(R/'candidate-resources.txt').read_text();regs=list(map(int,re.findall(r'REG:(\d+)',resources)))
assert len(regs)==20 and min(regs)==48 and max(regs)==61
assert all(x=='0' for x in re.findall(r'(?:LOCAL|STACK):(\d+)',resources))
assert counts==dict(adjacent_graph_pairs=6600,comparison_reference_checks=660,trace_replays=1080)
result=dict(production_sha256=sha(R/'previous.py'),candidate_sha256=before,**counts,fresh_native_graph_replays=240,suite_passes=16,suite_changed_native_cases=2,bench_shapes=12,assembly_variants=20,no_clock_sass_identity_both_pdl_modes=True,all_coherent_loads_after_wait=True,wait_probe_brackets_only_acqbulk=True,consecutive_clock_probe_has_no_intervening_instructions=True,register_range=[48,61],trace_indexing_injective=True,judge_requests=0,sanitizer='not run; trace index audit only',limitations='Separate probes alter scheduling/registers; do not add interval medians, infer production phase percentages, or subtract clock-control graph costs.')
(R/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

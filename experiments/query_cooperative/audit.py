"""CPU reproducibility, layout, unchanged-function, and result-count audit."""
import ast,hashlib,json,re,statistics,runpy,csv,io
from pathlib import Path
from generate import generate
ROOT=Path(__file__).resolve().parent
R=ROOT/'results'
s=(ROOT/'production_snapshot.py').read_text()
v=(ROOT/'candidate.py').read_text()
sw=(ROOT/'swizzled.py').read_text()
dist=(ROOT/'distributed.py').read_text()
assert generate(s)==v
assert sw==v.replace('muse_native_query_cooperative_s33','muse_native_query_cooperative_swizzle_s33').replace('Qpacked[qhead*16+lane%16]','Qpacked[qhead*16+((lane%16)^((qhead/2)*4))]').replace('Qpacked[(lane/4)*16+d*8+(lane%4)+i*4]','Qpacked[(lane/4)*16+((d*8+(lane%4)+i*4)^((lane/8)*4))]')
assert (ROOT.parent.parent/'submission.py').read_text()==s
manifest=json.loads((R/'source-manifest.json').read_text())
for name,digest in manifest.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
runpy.run_path(str(ROOT/'generate.py'),run_name='__main__')
assert json.loads((R/'source-manifest.json').read_text())==manifest
def defs(text):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(text)) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
assert defs(s)==defs(v)==defs(sw)==defs(dist)

# Distributed matrix rows: two real heads in each 16-row WGMMA warp block.
# Shuffle sources hold exactly the desired four-column FP8 word; output stores
# cover every (head,column) once and ignore the padded matrix rows.
outputs=[]
for warp in range(4):
    for lane in range(8):
        for d in range(2):
            for i in range(2):
                src=((lane//4)*16+d*8+lane%4+i*4)&31
                assert warp*2+src//16==warp*2+lane//4
                assert (src%16)*4==d*32+(lane%4)*4+i*16
        for i in range(32):
            if not i&2:
                outputs.append((warp*2+lane//4,(i//8)*16+(lane%4)*2+(i&1)+(i&4)*2))
assert sorted(outputs)==[(h,c) for h in range(8) for c in range(64)]

# Each packed word has exactly one producer and one consumer. All four 32-lane
# store instructions and each of warp zero's four load instructions are bank
# conflict free with the XOR swizzle. The linear layout has four-way read banks.
layout={}
for kind in ('linear','swizzled'):
    def address(head,col):return head*16+(col^((head//2)*4) if kind=='swizzled' else col)
    writers=[];readers=[];write_conflicts=[];read_conflicts=[]
    for warp in range(4):
        addresses=[address(warp*2+lane//16,lane%16) for lane in range(32)]
        writers+=addresses
        write_conflicts.append(max(addresses.count(a) for a in addresses))
        banks=[a%32 for a in addresses]
        assert len(set(banks))==32
    for d in range(2):
        for i in range(2):
            addresses=[address(lane//4,d*8+lane%4+i*4) for lane in range(32)]
            readers+=addresses;banks=[a%32 for a in addresses]
            read_conflicts.append(max(banks.count(b) for b in banks))
    assert sorted(writers)==sorted(readers)==list(range(128))
    assert read_conflicts==([1]*4 if kind=='swizzled' else [4]*4)
    layout[kind]=dict(packed_words=128,read_bank_multiplicity=read_conflicts)

counts={};summary={}
for name,records,pairs in [('initial',32,60),('swizzled-initial',32,60),('swizzled-confirmation',16,120),('linear-confirmation',8,120),('distributed-initial',32,60),('distributed-confirmation',16,120)]:
    rows=json.loads((R/(name+'.json')).read_text());assert len(rows)==records
    for row in rows:
        raw=row['raw'];assert len(raw)==pairs
        assert abs(row['paired_delta_us']-statistics.median(x[1]-x[0] for x in raw))<1e-9
    text=(R/(name+'.log')).read_text()
    assert text.count('PASS ')==records//2*9
    counts[name]=dict(pairs=records*pairs,reference_checks=records//2*9)
    summary[name]=[{k:v for k,v in row.items() if k!='raw'} for row in rows]
for name in ('fresh','swizzled-fresh','distributed-fresh'):
    rows=json.loads((R/(name+'.json')).read_text());assert len(rows)==16
    assert sum(row['replays'] for row in rows)==64
    assert (R/(name+'.log')).read_text().count('PASS ')==64
    counts[name]=64
for name in ('suite','swizzled-suite','distributed-suite'):
    text=((R/'distributed-test.log').read_text()+(R/'distributed-bench.log').read_text()) if name=='distributed-suite' else (R/(name+'.log')).read_text()
    assert sum(line.startswith('PASS ') for line in text.splitlines())==16
    assert len(re.findall(r'^(?:decode|verify)\s+(?:1|16|32)\s+(?:sliding|full)\s+',text,re.M))==12
    counts[name]=dict(tests=16,bench_shapes=12)
for tool in ('memcheck','synccheck','racecheck','distributed-memcheck','distributed-synccheck','distributed-racecheck'):
    text=(R/(tool+'.log')).read_text()
    assert text.count('PASS native-only')==24
    assert ('ERROR SUMMARY: 0 errors' in text if not tool.endswith('racecheck') else 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' in text)
    counts[tool]=24

# Resource/SASS evidence is static. No inference of achieved occupancy or cycle
# savings from instruction counts alone. Include opcode counts for all variants.
sass={}
for label in ('parent','candidate','swizzled','distributed'):
    text=(R/(label+'.sass.txt')).read_text();blocks=re.findall(r'Function : (\S+)\n(.*?)(?=Function :|\Z)',text,re.S)
    sass[label]={}
    for name,body in blocks:
        ins=re.findall(r'/\*[0-9a-f]+\*/\s+([^\n]*?;)',body)
        if 'sliding' not in name:continue
        ops={op:sum(op in x for x in ins) for op in ('LDS','STS','BAR.SYNC','QGMMA','HGMMA','LDG','SHFL')}
        assert ops['BAR.SYNC']==3 and ops['QGMMA']==2 and ops['HGMMA']==4
        if label=='distributed':
            idx=[x for x in ins if 'SHFL.IDX' in x]
            assert len(idx)==4 and all('@' not in x for x in idx)
        sass[label][name]=dict(instructions=len(ins),opcodes=ops)
    assert len(sass[label])==4
    resources=(R/(label+'.resources.txt')).read_text()
    assert resources.count('REG:74 STACK:0')==4
    assert resources.count('LOCAL:0')==4
    assert resources.count('SHARED:'+('17408' if label in ('parent','distributed') else '17920'))==4
profiles={}
for label in ('parent','linear','swizzled','distributed'):
    text=(R/('ncu-'+label+'.log')).read_text()
    assert text.count("PASS ('profile'")==1
    start=text.index('"ID","Process ID"')
    profiles[label]={row['Metric Name']:float(row['Metric Value'].replace(',','')) for row in csv.DictReader(io.StringIO(text[start:]))}
assert profiles['parent']['l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum']==0
assert profiles['linear']['l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum']==768
assert profiles['swizzled']['l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum']==0
assert profiles['distributed']['l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum']==0
report=dict(production_sha256=hashlib.sha256(s.encode()).hexdigest(),shared_functions=len(defs(s)),layout=layout,counts=counts,sass=sass,profiles=profiles,summary=summary)
(R/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('sass','summary')},indent=2))

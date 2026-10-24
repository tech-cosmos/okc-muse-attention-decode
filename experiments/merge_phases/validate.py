"""Offline, fresh-input graph diagnostics. No data enters production from this driver."""
import fcntl,importlib.util,json,statistics,sys,gzip
from pathlib import Path
sys.path.insert(0,'.')
import torch,reference
ROOT=Path(__file__).resolve().parent
mods={}
for name in ('previous','candidate'):
    spec=importlib.util.spec_from_file_location('s39'+name,ROOT/(name+'.py'))
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);mods[name]=m
old,probe=mods.values()
cases=[dict(batch=1,ctx=80000,heads='tp8',layer='full',q_len=l,page=p,dist='coherent') for p in (1,64) for l in (1,8)]
cases += [dict(batch=2,ctx=9000,heads='tp8',layer='full',q_len=8,page=p,dist='edges',kv_scale=1/64) for p in (1,64)]
labels=['no_clock','prologue','pdl_wait','stats_weights','partial_load_fma','publication_barrier','crosswarp','norm_gate_store','whole','clock_control']
def check(d,o,label):
    torch.cuda.synchronize();ok,msg=reference.check_implementation(d,reference.collect(d,o));assert ok,(label,msg)
    print('PASS',label,flush=True)
def capture(fn):
    for _ in range(3):o=fn()
    torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):o=fn()
    return g,o

def compare():
    confirm=sys.argv[1]=='confirm';seed=39072 if confirm else 39071
    selected=cases if not confirm else [cases[i] for i in (0,1,4,5)]
    flush=torch.empty(512<<20,device='cuda',dtype=torch.uint8);x=torch.zeros(1,device='cuda');eg,_=capture(lambda:x.add_(1))
    a,b=[torch.cuda.Event(enable_timing=True) for _ in range(2)]
    def time(g):
        flush.zero_();a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000
    results=[]
    for i,c in enumerate(selected):
        d=reference.generate_input(seed=seed,**c)
        for phase in list(range(10))+[-1]:
            probe.PHASE=max(0,phase)
            functions={'previous':lambda:old.kernel_fn(d),'probe':lambda:(old if phase==-1 else probe).kernel_fn(d)}
            graphs={};outs={};trace=None
            for name in (list(functions)[::-1] if confirm else functions):
                check(d,functions[name](),(i,phase,name,'eager'))
                graphs[name],outs[name]=capture(functions[name]);graphs[name].replay();check(d,outs[name],(i,phase,name,'capture'))
                if name=='probe' and phase>0:trace=probe.TRACE
            raw=[]
            for r in range(60):
                order=['previous','probe'] if r%2==0 else ['probe','previous']
                ts={name:time(graphs[name]) for name in order};raw.append([ts['previous'],ts['probe'],time(eg)])
            traces=[]
            if phase>0:
                for _ in range(12):
                    flush.zero_();graphs['probe'].replay();torch.cuda.synchronize();traces.append(trace.cpu().tolist())
            for name in graphs:graphs[name].replay();check(d,outs[name],(i,phase,name,'after'))
            cost=statistics.median(x[2] for x in raw)
            warp=[]
            if phase>0:
                for w in range(8):
                    vals=[row[w][1]-row[w][0] for t in traces for row in t if row[w][1]!=0]
                    warp.append(statistics.median(vals) if vals else None)
            row=dict(shape=c,phase=phase,label=labels[phase] if phase>=0 else 'identical_capture',previous_us=statistics.median(x[0] for x in raw)-cost,probe_us=statistics.median(x[1] for x in raw)-cost,paired_delta_us=statistics.median(x[1]-x[0] for x in raw),warp_cycles=warp,raw=raw,traces=traces)
            results.append(row);print(json.dumps({k:v for k,v in row.items() if k not in ('raw','traces')}),flush=True)
            (ROOT/('pairs-'+sys.argv[1]+'.json.gz')).write_bytes(gzip.compress(json.dumps(results).encode(),compresslevel=1,mtime=0))
            (ROOT/('summary-'+sys.argv[1]+'.json')).write_text(json.dumps([{k:v for k,v in x.items() if k not in ('raw','traces')} for x in results],indent=2)+'\n')
            del graphs,outs,trace;torch.cuda.empty_cache()
        del d;torch.cuda.empty_cache()
def fresh():
    results=[]
    for i,c in enumerate(cases):
        d=reference.generate_input(seed=39081,**c);graphs={};outs={};traces={}
        for phase in range(10):
            probe.PHASE=phase;graphs[phase],outs[phase]=capture(lambda:probe.kernel_fn(d));traces[phase]=probe.TRACE
        for rep in range(4):
            n=reference.generate_input(seed=39082+rep,**c)
            for key,v in d.items():
                if isinstance(v,torch.Tensor):v.copy_(n[key])
            if rep>=2:
                if rep==3:d['seq_lens'].zero_()
                else:d['seq_lens'].div_(2,rounding_mode='floor')
                p=d['seq_lens'].long()[:,None]+torch.arange(c['q_len'],device='cuda')[None,:]
                d['positions'].copy_(p.flatten());d['cache_seqlens'].copy_(d['seq_lens']+c['q_len'])
                d['cache_loc'].copy_((torch.gather(d['page_table'].long(),1,p//c['page'])*c['page']+p%c['page']).flatten())
            for phase in graphs:graphs[phase].replay();check(d,outs[phase],('fresh',i,phase,rep))
        results.append(dict(shape=c,replays=40));(ROOT/'fresh.json').write_text(json.dumps(results,indent=2)+'\n')
        del d,n,graphs,outs,traces;torch.cuda.empty_cache()
if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='suite':
        import tools.bench as bench
        probe.PHASE=8;bench.submission=probe
        with open('/tmp/okc-gpu.lock','w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);assert bench.test()==0
        bench.bench()
    else:
        with open('/tmp/okc-gpu.lock','w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);globals()['compare' if mode=='confirm' else mode]()

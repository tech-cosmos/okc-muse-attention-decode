"""Offline sequential graph reuse and generation-wrap validation; no timing claim."""
import fcntl, importlib.util, json, sys
from pathlib import Path
sys.path.insert(0,'.')
import torch, reference
R=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('s42countermerged',R/'merged.py')
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
def run():
    graphs=[]
    for batch in (12,16,32):
        c=dict(batch=batch,ctx=6000,heads='tp8',layer='full',q_len=8,page=64,dist='coherent',spread='mixed')
        d=reference.generate_input(seed=42101,**c)
        p=m._plan(d['config'],d['page_table'].shape[1],d['qkv'].device)
        assert p['gluon'] and p['gfuse'] and p['splits']*batch <= 2*m._num_sms(d['qkv'].device)
        for _ in range(3):m.kernel_fn(d)
        torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):o=m.kernel_fn(d)
        graphs.append((c,d,g,o))
    buf=m._counters(32,torch.device('cuda:0'),'g')
    records=[]
    for generation in (0,0x7ffe,0xfffe):
        word=generation<<16
        if word>=2**31:word-=2**32
        buf.fill_(word);expected=[generation]*32
        for rep in range(4):
            order=graphs if rep%2==0 else graphs[::-1]
            for c,d,g,o in order:
                n=reference.generate_input(seed=42102+rep,**c)
                for k,v in d.items():
                    if isinstance(v,torch.Tensor):v.copy_(n[k])
                g.replay();torch.cuda.synchronize()
                ok,msg=reference.check_implementation(d,reference.collect(d,o));assert ok,msg
                for i in range(c['batch']):expected[i]=(expected[i]+1)&0xffff
                actual=[int(v)&0xffffffff for v in buf[:32].cpu().tolist()]
                assert actual==[v<<16 for v in expected],(generation,rep,c['batch'],actual,expected)
                row=dict(initial_generation=generation,round=rep,batch=c['batch'],generations=expected.copy(),output_cache_pass=True)
                records.append(row);print('PASS',generation,rep,c['batch'],flush=True)
    (R/'counters.json').write_text(json.dumps(records,indent=2)+'\n')
if __name__=='__main__':
    with open('/tmp/okc-gpu.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);run()

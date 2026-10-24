"""Validate late main 9a43323 with native merger; cold complete-graph pairs."""
import fcntl
import importlib.util
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, '.')
import torch
import reference

ROOT = Path(__file__).resolve().parent
MODE = sys.argv[1]
mods = {}
for name in (('integrated', 'upstream', 'previous') if MODE == 'confirm'
             else ('previous', 'upstream', 'integrated')):
    spec = importlib.util.spec_from_file_location('s43d1_'+name, ROOT/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    mods[name] = module

def case(ctx=80000, page=1, scale=1., dist='coherent', **kwargs):
    return dict(batch=1, ctx=ctx, heads='tp8', layer='full', q_len=1,
                page=page, kv_scale=scale, dist=dist, **kwargs)

CASES = [case(), case(page=64), case(10, dist='normal'),
         case(1000, 64, dist='edges'), case(4095, scale=1/64, dist='soft'),
         case(4096, scale=1/64, dist='soft'), case(8192, 64, 1/64, 'edges'),
         case(33333, 64, 1/64, 'soft'), case(120000)]
CASES += [{**case(), 'q_len': 8}, {**case(dist='soft'), 'layer': 'sliding'},
          {**case(9000, 64, 1/64, 'edges'), 'batch': 2, 'q_len': 8},
          {**case(5000, dist='normal'), 'heads': 'dp'}]

def check(data, out, label):
    torch.cuda.synchronize()
    ok, message = reference.check_implementation(data, reference.collect(data, out))
    assert ok, (label, message)
    print('PASS', label, flush=True)

def capture(fn):
    for _ in range(3):
        out = fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    return graph, out

def dispatch(module, data, native):
    cfg = data['config']
    plan = module._plan(cfg, data['page_table'].shape[1], data['qkv'].device)
    span = data['page_table'].shape[1]*cfg['page_size']
    if cfg['window']:
        span = min(span, cfg['window'])
    chunk = ((span+plan['splits']-1)//plan['splits']+plan['block_n']-1)//plan['block_n']*plan['block_n']
    splits = plan['splits'] if plan.get('balance') else (span+chunk-1)//chunk
    full = native and not cfg['window'] and cfg['batch']*cfg['num_kv_heads']<=4 and 64<splits<=128 and not plan.get('gluon')
    return dict(plan=plan, splits=splits, native_full=full)

def compare():
    cases = CASES if MODE == 'compare' else [CASES[i] for i in (0, 1, 4, 5, 6, 7, 8)]
    flush = torch.empty(512<<20, dtype=torch.uint8, device='cuda')
    empty_tensor = torch.zeros(1, device='cuda')
    empty_graph, _ = capture(lambda: empty_tensor.add_(1))
    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    def timed(graph):
        flush.zero_(); start.record(); graph.replay(); end.record(); end.synchronize()
        return start.elapsed_time(end)*1000
    results = []
    for i, shape in enumerate(cases):
        data = reference.generate_input(seed=43101 if MODE=='compare' else 43102, **shape)
        graphs, outs, plans = {}, {}, {}
        for name, module in mods.items():
            plans[name] = dispatch(module, data, name!='upstream')
            check(data, module.kernel_fn(data), (i, name, 'eager'))
            graphs[name], outs[name] = capture(lambda: module.kernel_fn(data))
            graphs[name].replay(); check(data, outs[name], (i, name, 'capture'))
        contrasts = {}
        for parent in ('previous', 'upstream'):
            raw = []
            for r in range(80):
                order = (parent, 'integrated') if r%2==0 else ('integrated', parent)
                times = {name: timed(graphs[name]) for name in order}
                raw.append([times[parent], times['integrated'], timed(empty_graph)])
            cost = statistics.median(e for _, _, e in raw)
            contrasts[parent] = dict(parent_us=statistics.median(a for a, _, _ in raw)-cost,
                                     integrated_us=statistics.median(b for _, b, _ in raw)-cost,
                                     paired_delta_us=statistics.median(b-a for a, b, _ in raw), raw=raw)
        for name in mods:
            graphs[name].replay(); check(data, outs[name], (i, name, 'after'))
        row = dict(shape=shape, dispatch=plans, contrasts=contrasts)
        results.append(row)
        print(json.dumps(dict(shape=shape, contrasts={k: {f:v for f,v in x.items() if f!='raw'}
                                                    for k,x in contrasts.items()})), flush=True)
        (ROOT/('pairs-'+MODE+'.json')).write_text(json.dumps(results, indent=2)+'\n')
        del data, graphs, outs
        torch.cuda.empty_cache()

def fresh():
    results = []
    for i, shape in enumerate(CASES):
        data = reference.generate_input(seed=43111, **shape)
        graphs, outs = {}, {}
        for name, module in mods.items():
            graphs[name], outs[name] = capture(lambda: module.kernel_fn(data))
        for rep in range(4):
            new = reference.generate_input(seed=43112+rep, **shape)
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    value.copy_(new[key])
            if rep>=2:
                if rep==3: data['seq_lens'].zero_()
                else: data['seq_lens'].div_(2, rounding_mode='floor')
                pos = data['seq_lens'].long()[:,None]+torch.arange(shape['q_len'],device='cuda')[None,:]
                data['positions'].copy_(pos.flatten())
                data['cache_seqlens'].copy_(data['seq_lens']+shape['q_len'])
                data['cache_loc'].copy_((torch.gather(data['page_table'].long(),1,pos//shape['page'])*shape['page']+pos%shape['page']).flatten())
            assert (data['cache_loc']>=0).all()
            for name in mods:
                graphs[name].replay(); check(data, outs[name], ('fresh',i,name,rep))
        results.append(dict(shape=shape, sources=list(mods), replays_per_source=4))
        (ROOT/'fresh.json').write_text(json.dumps(results,indent=2)+'\n')
        del data, new, graphs, outs
        torch.cuda.empty_cache()

if MODE=='suite':
    import tools.bench as bench
    bench.submission = mods['integrated']
    with open('/tmp/okc-gpu.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        assert bench.test()==0
    bench.bench()
else:
    with open('/tmp/okc-gpu.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        (compare if MODE in ('compare','confirm') else fresh)()

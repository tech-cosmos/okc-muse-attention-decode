"""Paired A/B over nearby decode/verify shapes: python tools/s18/abnear.py <dirA> <dirB> layer q_len batches(csv) [ctx]."""
import sys, os, importlib.util, statistics, torch
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(1, "/home/daytona/repo")
import bench, reference
mods = []
for i, d in enumerate(sys.argv[1:3]):
    spec = importlib.util.spec_from_file_location(f"sub_ab{i}", os.path.join(d, "submission.py"))
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m); mods.append(m)
layer, q_len = sys.argv[3], int(sys.argv[4])
ctx = int(sys.argv[6]) if len(sys.argv) > 6 else 80000
for batch in map(int, sys.argv[5].split(",")):
    c = dict(batch=batch, ctx=ctx, heads="tp8", layer=layer, page=1, q_len=q_len, dist="soft", spread="agentic")
    data = reference.generate_input(seed=11, **c)
    oks = []
    for m in mods:
        out = m.kernel_fn(data); torch.cuda.synchronize()
        oks.append(reference.check_implementation(data, reference.collect(data, out))[0])
    ts = [[], []]
    for r in range(5):
        for i, m in enumerate(mods):
            ts[i].append(bench.time_graph(lambda: m.kernel_fn(data)) - bench.empty_cost())
    a, b = statistics.median(ts[0]), statistics.median(ts[1])
    print(f"{layer} L={q_len} B={batch}: A {a:.2f}  B {b:.2f}  B-A {b - a:+.2f}  ok {oks}", flush=True)
    del data; torch.cuda.empty_cache()

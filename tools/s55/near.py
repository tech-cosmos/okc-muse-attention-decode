"""Paired A/B (alternating order) of two submission files on nearby verify-full shapes, with a correctness check.
near.py a.py b.py rounds"""
import importlib.util, statistics, sys
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench, reference, torch
def load(path, name):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
A, B = load(sys.argv[1], "sub_a"), load(sys.argv[2], "sub_b")
rounds = int(sys.argv[3])
for batch, ctx, page in ((1, 40000, 1), (1, 120000, 1), (1, 80000, 64), (2, 80000, 1), (1, 300000, 1)):
    c = dict(batch=batch, ctx=ctx, heads="tp8", layer="full", page=page, q_len=8, dist="coherent",
             **({"spread": "agentic"} if batch > 1 else {}))
    d = reference.generate_input(seed=7, **c)
    out = B.kernel_fn(d); torch.cuda.synchronize()
    ok, msg = reference.check_implementation(d, reference.collect(d, out))
    ta, tb = [], []
    for i in range(rounds):
        fa = lambda: bench.time_graph(lambda: A.kernel_fn(d)); fb = lambda: bench.time_graph(lambda: B.kernel_fn(d))
        if i % 2: tb.append(fb()); ta.append(fa())
        else: ta.append(fa()); tb.append(fb())
    df = [y - x for x, y in zip(ta, tb)]
    print(f"B{batch} ctx {ctx} page {page}: {'PASS' if ok else 'FAIL'} A {statistics.median(ta):7.2f} B {statistics.median(tb):7.2f} "
          f"paired {statistics.median(df):+.2f} B wins {sum(x < 0 for x in df)}/{rounds}", flush=True)

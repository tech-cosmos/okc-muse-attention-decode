"""Paired A/B of two submission files on nearby shapes: abn.py a.py b.py rounds layer q_len batch [batch...]"""
import importlib.util, statistics, sys
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench, reference, torch
def load(path, name):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
A, B = load(sys.argv[1], "sub_a"), load(sys.argv[2], "sub_b")
rounds, layer, q_len = int(sys.argv[3]), sys.argv[4], int(sys.argv[5])
for batch in map(int, sys.argv[6:]):
    c = dict(batch=batch, ctx=80000, heads="tp8", layer=layer, page=1, q_len=q_len,
             dist="soft" if layer == "sliding" else "coherent", **({"spread": "agentic"} if batch > 1 else {}))
    d = reference.generate_input(seed=7, **c)
    if batch <= 16:
        ok, msg = reference.check_implementation(d, reference.collect(d, B.kernel_fn(d)))
    else:
        ok, msg = True, "unchecked"
    ta, tb = [], []
    for _ in range(rounds):
        ta.append(bench.time_graph(lambda: A.kernel_fn(d)))
        tb.append(bench.time_graph(lambda: B.kernel_fn(d)))
    df = [y - x for x, y in zip(ta, tb)]
    print(f"{layer} L={q_len} B={batch:2d} A {statistics.median(ta):7.2f} B {statistics.median(tb):7.2f} paired "
          f"{statistics.median(df):+.2f} wins {sum(x < 0 for x in df)}/{rounds} {'ok' if ok else 'FAIL ' + msg[:60]}",
          flush=True)
    del d; torch.cuda.empty_cache()

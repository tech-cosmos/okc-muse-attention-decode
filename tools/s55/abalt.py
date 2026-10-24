"""Paired A/B of two submission files on scored shapes, alternating which runs first each round (cancels order bias).
abalt.py a.py b.py rounds 'decode 1 sliding' [...]"""
import importlib.util, statistics, sys
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench, torch
def load(path, name):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
A, B = load(sys.argv[1], "sub_a"), load(sys.argv[2], "sub_b")
rounds = int(sys.argv[3])
for shape in sys.argv[4:]:
    for ld, c in bench.SCORED:
        if f"{ld} {c['layer']}" != shape: continue
        d = bench.data_for(ld, c)
        ta, tb = [], []
        for i in range(rounds):
            if i % 2:
                tb.append(bench.time_graph(lambda: B.kernel_fn(d))); ta.append(bench.time_graph(lambda: A.kernel_fn(d)))
            else:
                ta.append(bench.time_graph(lambda: A.kernel_fn(d))); tb.append(bench.time_graph(lambda: B.kernel_fn(d)))
        df = [y - x for x, y in zip(ta, tb)]
        ev, od = df[0::2], df[1::2]
        print(f"{shape:18s} A {statistics.median(ta):7.2f} B {statistics.median(tb):7.2f} paired {statistics.median(df):+.2f} "
              f"(A-first {statistics.median(ev):+.2f}, B-first {statistics.median(od):+.2f}) B wins {sum(x<0 for x in df)}/{rounds}", flush=True)

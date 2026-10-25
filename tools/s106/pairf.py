"""s106: paired H100 timing (cold L2) of two submission files on scored shapes. pairf.py a.py b.py rounds 'shape' ..."""
import importlib.util, statistics, sys
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench
def load(path, name):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
A, B = load(sys.argv[1], "sub_a"), load(sys.argv[2], "sub_b")
rounds = int(sys.argv[3])
for shape in sys.argv[4:]:
    load_, c = [x for x in bench.SCORED if f"{x[0]} {x[1]['layer']}" == shape][0]
    d = bench.data_for(load_, c)
    ta, tb = [], []
    for r in range(rounds):
        t = {}
        for m in ((A, B) if r % 2 == 0 else (B, A)): t[m] = bench.time_graph(lambda: m.kernel_fn(d)) - bench.empty_cost()
        ta.append(t[A]); tb.append(t[B])
    df = [y - x for x, y in zip(ta, tb)]
    print(f"{shape:18s} A {statistics.median(ta):7.2f} B {statistics.median(tb):7.2f} paired {statistics.median(df):+.2f} wins {sum(x < 0 for x in df)}/{rounds}", flush=True)

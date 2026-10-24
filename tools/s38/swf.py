"""Time plan overrides on a scored shape for any submission file: swf.py file.py "<shape>" "<json>" ..."""
import importlib.util, json, statistics, sys
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench
sp = importlib.util.spec_from_file_location("subx", sys.argv[1]); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
shape = sys.argv[2]
for ld, c in bench.SCORED:
    if f"{ld} {c['layer']}" != shape: continue
    d = bench.data_for(ld, c)
    orig = m._plan
    for a in sys.argv[3:]:
        p = json.loads(a)
        m._plan = lambda cfg, mp, dev: {**orig(cfg, mp, dev), **p}
        ts = [bench.time_graph(lambda: m.kernel_fn(d)) - bench.empty_cost() for _ in range(3)]
        m._plan = orig
        print(f"{shape} {statistics.median(ts):7.2f} {p}", flush=True)

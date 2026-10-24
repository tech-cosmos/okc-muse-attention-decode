"""Run a submission file 3x on a scored shape (for ncu): one.py file.py "<shape>" ["<json plan>"]"""
import importlib.util, json, sys, torch
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench
sp = importlib.util.spec_from_file_location("subx", sys.argv[1]); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
shape = sys.argv[2]
p = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
orig = m._plan
m._plan = lambda cfg, mp, dev: {**orig(cfg, mp, dev), **p}
for ld, c in bench.SCORED:
    if f"{ld} {c['layer']}" != shape: continue
    d = bench.data_for(ld, c)
    for _ in range(3):
        m.kernel_fn(d)
    torch.cuda.synchronize()

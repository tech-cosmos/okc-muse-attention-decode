"""s103: outputs of submission files vs the first one on scored shapes (max abs / rel L2 diff)."""
import importlib.util, sys, torch
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench
def load(path, name):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
mods = [load(p, f"m{i}") for i, p in enumerate(sys.argv[1].split(","))]
for shape in sys.argv[2:]:
    ld, c = [x for x in bench.SCORED if f"{x[0]} {x[1]['layer']}" == shape][0]
    d = bench.data_for(ld, c)
    outs = []
    for m in mods:
        for _ in range(3): o = m.kernel_fn(d).float()
        torch.cuda.synchronize(); outs.append(o.clone())
    for i, o in enumerate(outs[1:], 1):
        df = (o - outs[0]); print(shape, i, "max abs", df.abs().max().item(), "rel L2", (df.norm() / outs[0].norm()).item(), flush=True)

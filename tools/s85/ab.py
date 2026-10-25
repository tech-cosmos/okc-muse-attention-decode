"""s85: paired H100 A/B of /tmp/main_sub.py (main) vs submission.py on chosen scored shapes.
  python tools/s85/ab.py "decode 1:sliding,verify 1:sliding" [rounds]"""
import sys, importlib.util, statistics, torch
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench, submission
spec = importlib.util.spec_from_file_location("main_sub", "/tmp/main_sub.py"); ms = importlib.util.module_from_spec(spec); spec.loader.exec_module(ms)
want = [tuple(x.split(":")) for x in sys.argv[1].split(",")]
R = int(sys.argv[2]) if len(sys.argv) > 2 else 9
for load, c in bench.SCORED:
    if (load, c["layer"]) not in want: continue
    data = bench.data_for(load, c)
    a = ms.kernel_fn(data).float(); b = submission.kernel_fn(data).float(); torch.cuda.synchronize()
    print(load, c["layer"], "rel diff vs main", ((a - b).norm() / a.norm()).item())
    d = []
    for r in range(R):
        ta = bench.time_graph(lambda: ms.kernel_fn(data)); tb = bench.time_graph(lambda: submission.kernel_fn(data))
        d.append(tb - ta)
    print(f"{load} {c['layer']}: main {ta - bench.empty_cost():.2f} new {tb - bench.empty_cost():.2f}; paired median diff {statistics.median(d):+.3f} us, wins {sum(x < 0 for x in d)}/{R}", flush=True)

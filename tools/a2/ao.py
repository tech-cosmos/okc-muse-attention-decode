"""agent2: time a scored shape with and without the merge launch (_combine_hb no-op), for plan overrides.
  python tools/a2/ao.py "verify 32 sliding" '[{}, {"splits": 8, "sb": 8}]' [reps]"""
import sys, json, statistics
sys.path.insert(0, "/home/daytona/repo2/tools"); sys.path.insert(0, "/home/daytona/repo2")
import bench, submission
name, plans = sys.argv[1], json.loads(sys.argv[2])
R = int(sys.argv[3]) if len(sys.argv) > 3 else 5
orig = submission._combine_hb
class Nop:
    def __getitem__(self, g): return lambda *a, **k: None
for load, c in bench.SCORED:
    if f"{load} {c['layer']}" != name: continue
    for p in plans:
        res = []
        for nop in (False, True):
            submission._combine_hb = Nop() if nop else orig
            res.append(statistics.median(bench.run_shape(load, c, p) for _ in range(R)))
        submission._combine_hb = orig
        print(f"{name} {p}: whole {res[0]:.2f} attn-only {res[1]:.2f} merge {res[0]-res[1]:.2f}", flush=True)

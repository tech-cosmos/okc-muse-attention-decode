import sys, os, json; sys.path.insert(0,'/home/daytona/repo/tools'); sys.path.insert(0,'/home/daytona/repo')
import submission, bench, statistics
class Noop:
    def __getitem__(self, g): return lambda *a, **k: None
if os.environ.get("NOMERGE"):
    submission._combine = Noop(); submission._combine_hb = Noop()
P=json.loads(sys.argv[2])
for load, c in bench.SCORED:
    if f"{load} {c['layer']}" in sys.argv[1].split(","):
        res=[[bench.run_shape(load, c, p) for p in P] for _ in range(3)]
        print(os.environ.get("NOMERGE",""), load, c['layer'], *[round(statistics.median(x),2) for x in zip(*res)], flush=True)

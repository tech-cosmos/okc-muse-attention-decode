"""Time a scored shape with plan overrides, whole call and attention only (merge kernels replaced by no-ops).
attn_only.py '<shape>' rounds '<json A>' ['<json B>' ...]"""
import json, statistics, sys
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench, submission


class _Noop:
    def __getitem__(self, grid):
        return lambda *a, **k: None


shape, rounds = sys.argv[1], int(sys.argv[2])
plans = [json.loads(a) for a in sys.argv[3:]]
orig = {n: getattr(submission, n) for n in ("_combine", "_combine_hb", "_combine_bal")}
for load, c in bench.SCORED:
    if f"{load} {c['layer']}" != shape:
        continue
    for i, p in enumerate(plans):
        tw, ta = [], []
        for _ in range(rounds):
            tw.append(bench.run_shape(load, c, p))
            for n in orig:
                setattr(submission, n, _Noop())
            try:
                ta.append(bench.run_shape(load, c, p))
            finally:
                for n, f in orig.items():
                    setattr(submission, n, f)
        print(f"{shape} whole {statistics.median(tw):7.2f} attn {statistics.median(ta):7.2f} {p}", flush=True)

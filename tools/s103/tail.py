"""s103: merge-tail size on full layers (H100): whole call vs first kernel alone (merge kernels replaced by no-ops)."""
import sys; sys.path.insert(0, "tools"); sys.path.insert(0, ".")
import bench, submission
class _Noop:
    def __getitem__(self, g): return lambda *a, **k: None
names = ("_red_fin", "_combine", "_combine_bal", "_combine_hb")
orig = {n: getattr(submission, n) for n in names}
for shape in sys.argv[1:]:
    load, c = [x for x in bench.SCORED if f"{x[0]} {x[1]['layer']}" == shape][0]
    data = bench.data_for(load, c)
    def t(): return bench.time_graph(lambda: submission.kernel_fn(data)) - bench.empty_cost()
    for rnd in range(3):
        w = t()
        for n in names: setattr(submission, n, _Noop())
        f = t()
        for n, fn in orig.items(): setattr(submission, n, fn)
        print(f"{shape:16s} whole {w:7.2f} first-only {f:7.2f} tail {w - f:5.2f}", flush=True)

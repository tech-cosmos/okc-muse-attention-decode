"""s101: decode 1 full decomposition on the H100: whole call vs _gd_full alone vs merge alone."""
import sys; sys.path.insert(0, "tools"); sys.path.insert(0, ".")
import bench, submission
load, c = [x for x in bench.SCORED if x[0] == sys.argv[1] and x[1]["layer"] == sys.argv[2]][0]
data = bench.data_for(load, c)
import os, torch
if os.environ.get("WARM"): bench._FLUSH = torch.empty(8, dtype=torch.uint8, device="cuda")
class _Noop:
    def __getitem__(self, g): return lambda *a, **k: None
orig = {n: getattr(submission, n) for n in ("_red_fin", "_combine", "_gd_full", "_attn_split", "_combine_hb")}
def t(): return bench.time_graph(lambda: submission.kernel_fn(data)) - bench.empty_cost()
for rnd in range(3):
    res = {"whole": t()}
    for n in ("_red_fin", "_combine", "_combine_hb"):
        setattr(submission, n, _Noop())
    res["first only"] = t()
    for n, f in orig.items(): setattr(submission, n, f)
    for n in ("_gd_full", "_attn_split"):
        setattr(submission, n, _Noop())
    res["merge only"] = t()
    for n, f in orig.items(): setattr(submission, n, f)
    print(rnd, {k: round(v, 2) for k, v in res.items()}, flush=True)

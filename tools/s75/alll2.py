"""Verify/decode sliding: real vs all-L2 (page_table % 4096: rows spread over L2 slices) attention-only and whole.
python tools/s75/alll2.py"""
import sys, statistics
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import torch, bench, submission


class _Noop:
    def __getitem__(self, grid):
        return lambda *a, **k: None


names = ("_combine", "_combine_hb", "_combine_bal")
orig = {n: getattr(submission, n) for n in names}
for load, c in bench.SCORED:
    if c["layer"] != "sliding" or c["batch"] == 1:
        continue
    d = bench.data_for(load, c)
    d2 = dict(d)
    d2["page_table"] = d["page_table"] % 4096
    res = []
    for tag, dd in (("real", d), ("allL2", d2)):
        for mode in ("whole", "attn"):
            if mode == "attn":
                for n in names:
                    setattr(submission, n, _Noop())
            try:
                t = statistics.median(bench.time_graph(lambda: submission.kernel_fn(dd)) - bench.empty_cost() for _ in range(3))
            finally:
                for n, f in orig.items():
                    setattr(submission, n, f)
            res.append(f"{tag} {mode} {t:6.2f}")
    print(load, "sliding:", " | ".join(res), flush=True)

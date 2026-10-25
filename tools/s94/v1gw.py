"""Verify 1 full (H100): main Triton plan vs Gluon _gw_verify (non-bal), whole and with kernels dropped (timing only)."""
import sys, json, os; sys.path.insert(0, "tools"); sys.path.insert(0, ".")
import submission; import bench
load, c = [x for x in bench.SCORED if x[0] == (sys.argv[2] if len(sys.argv) > 2 else "verify 1") and x[1]["layer"] == "full"][0]
class Nop:
    def __getitem__(self, g): return lambda *a, **k: None
GW = dict(gluon=True, splits=127, block_n=64, stages=3, lag=1, gpeel=True, pw=4, pregs=40, cregs=96, maxnreg=80,
          lsum=True, gfuse=False, gbal=False)
plans = json.loads(sys.argv[1]) if len(sys.argv) > 1 else [{}, GW]
orig = {k: getattr(submission, k) for k in ("_combine", "_new_tokens", "_combine_hb")}
def drop(*names):
    for k, v in orig.items():
        setattr(submission, k, Nop() if k in names else v)
for rnd in range(3):
    for p in plans:
        p = {**GW, **p} if p.get("gluon") else p
        r = []
        for names in [(), ("_new_tokens",), ("_combine", "_combine_hb"), ("_new_tokens", "_combine", "_combine_hb")]:
            drop(*names)
            r.append(bench.run_shape(load, c, p))
        drop()
        print(rnd, {k: v for k, v in p.items() if k in ("gluon", "splits", "stages", "block_n")},
              "whole %.2f  -newtok %.2f  -merge %.2f  attn %.2f" % tuple(r), flush=True)

"""agent2: s75 proxy with fewer splits: verify blocks over ~2048 cached keys on a FULL layer (no RoPE/window),
Gluon _gw_verify (non-bal) at several splits vs the Triton verify-sliding plan. python tools/a2/proxy2.py"""
import sys, statistics
sys.path.insert(0, "/home/daytona/repo2/tools"); sys.path.insert(0, "/home/daytona/repo2")
import torch, bench, reference, submission
for batch in (16, 32):
    c = dict(batch=batch, ctx=2048, heads="tp8", layer="full", page=1, q_len=8, dist="soft")
    d = reference.generate_input(seed=99, **c)
    tri = dict(splits=4 if batch == 32 else 8, block_n=64, warps=4, stages=5, hpp=1, fuse=False, qk8=True, hb=2,
               sb=4 if batch == 32 else 8, cw=1, pdt="f32", peel=False, mif=True, gluon=False, gbal=False, gdec=False)
    plans = {"triton-sliding-plan": tri}
    for sp in (4, 8):
        for gf in (True, False):
            plans[f"gluon s{sp} gfuse{int(gf)}"] = {"gbal": False, "splits": sp, "gfuse": gf}
    orig = submission._plan
    for name, p in plans.items():
        ts = []
        for _ in range(3):
            submission._plan = lambda cfg, mp, dev, p=p: {**orig(cfg, mp, dev), **p}
            try:
                ts.append(bench.time_graph(lambda: submission.kernel_fn(d)) - bench.empty_cost())
            finally:
                submission._plan = orig
        print(f"batch {batch} ctx 2048 full verify: {name:22s} {statistics.median(ts):6.2f} us", flush=True)

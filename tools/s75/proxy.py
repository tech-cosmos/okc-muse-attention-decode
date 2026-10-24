"""Proxy: verify blocks over ~2048 cached keys on a FULL layer (no RoPE, no window), Gluon gw_verify plans vs the
Triton plan that verify sliding uses. python tools/s75/proxy.py"""
import sys, statistics
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import torch, bench, reference, submission
for batch in (16, 32):
    c = dict(batch=batch, ctx=2048, heads="tp8", layer="full", page=1, q_len=8, dist="soft", spread="agentic")
    d = reference.generate_input(seed=99, **c)
    tri = dict(splits=4 if batch == 32 else 8, block_n=64, warps=4, stages=5, hpp=1, fuse=False, qk8=True, hb=2,
               sb=4 if batch == 32 else 8, cw=1, pdt="f32", peel=True, gluon=False, gbal=False, gdec=False)
    plans = {"gluon(main)": None, "gluon nobal": {"gbal": False}, "triton-sliding-plan": tri}
    orig = submission._plan
    for name, p in plans.items():
        ts = []
        for _ in range(3):
            if p:
                submission._plan = lambda cfg, mp, dev, p=p: {**orig(cfg, mp, dev), **p}
            try:
                ts.append(bench.time_graph(lambda: submission.kernel_fn(d)) - bench.empty_cost())
            finally:
                submission._plan = orig
        print(f"batch {batch} ctx 2048 full verify: {name:22s} {statistics.median(ts):6.2f} us", flush=True)
for load, c in bench.SCORED:
    if c["layer"] == "sliding" and c["q_len"] == 8 and c["batch"] > 1:
        print(load, "sliding (real, main):", f"{bench.run_shape(load, c):.2f}")

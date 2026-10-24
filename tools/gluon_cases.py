"""Correctness on shapes that reach the Gluon verify-full path (batch >= 6 pairs, q_len 8, full layer), which the
judge's 16 tests don't: page 1/64, kv_scale, spreads, dists, and batches where the merge plan changes."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
import reference
import submission

CASES = [
    dict(batch=12, ctx=80000, heads="tp8", layer="full", q_len=8, page=1, spread="agentic", dist="coherent"),
    dict(batch=16, ctx=80000, heads="tp8", layer="full", q_len=8, page=1, spread="agentic", dist="coherent"),
    dict(batch=32, ctx=80000, heads="tp8", layer="full", q_len=8, page=1, spread="agentic", dist="soft"),
    dict(batch=64, ctx=40000, heads="tp8", layer="full", q_len=8, page=1, spread="mixed", dist="coherent"),
    dict(batch=16, ctx=20000, heads="tp8", layer="full", q_len=8, page=64, kv_scale=0.015625, dist="edges"),
    dict(batch=24, ctx=9000, heads="tp8", layer="full", q_len=8, spread="mixed", dist="normal"),
    dict(batch=8, ctx=30000, heads="tp8", layer="full", q_len=8, page=1, dist="coherent"),
    dict(batch=20, ctx=6000, heads="tp8", layer="full", q_len=8, page=64, dist="edges"),
]
bad = 0
for i, c in enumerate(CASES):
    data = reference.generate_input(seed=777 + i, **c)
    out = submission.kernel_fn(data)
    torch.cuda.synchronize()
    ok, msg = reference.check_implementation(data, reference.collect(data, out))
    bad += not ok
    print(f"{'PASS' if ok else 'FAIL'} {c} plan={submission._plan(data['config'], data['page_table'].shape[1], out.device)} {msg[:70]}", flush=True)
    del data
    torch.cuda.empty_cache()
sys.exit(bad)

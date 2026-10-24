"""Correctness on shapes that reach the batch*hkv <= 2 decode full-layer plan (not all in the judge's 16 tests)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import torch, reference, submission
CASES = [dict(batch=1, ctx=80000, heads="tp8", layer="full", page=1, dist="coherent"),
         dict(batch=1, ctx=10, heads="tp8", layer="full"),
         dict(batch=1, ctx=1000, heads="tp8", layer="full", page=64, dist="edges"),
         dict(batch=1, ctx=33333, heads="tp8", layer="full", kv_scale=0.015625, dist="soft"),
         dict(batch=2, ctx=20000, heads="dp", layer="full", dist="soft"),
         dict(batch=1, ctx=5000, heads="dp", layer="full", page=1, dist="normal")]
bad = 0
for i, c in enumerate(CASES):
    data = reference.generate_input(seed=4242 + i, **c)
    out = submission.kernel_fn(data); torch.cuda.synchronize()
    ok, msg = reference.check_implementation(data, reference.collect(data, out))
    bad += not ok
    print("PASS" if ok else "FAIL", c, submission._plan(data["config"], data["page_table"].shape[1], out.device).get("hpp"), msg[:60], flush=True)
sys.exit(bad)

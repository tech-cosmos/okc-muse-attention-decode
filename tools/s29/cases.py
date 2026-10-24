"""Correctness of a plan override on decode-1/2 full cases. python tools/s29/cases.py '<json override>'"""
import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import torch, reference, submission
over = json.loads(sys.argv[1])
orig = submission._plan
def plan(cfg, mp, dev):
    p = orig(cfg, mp, dev)
    return {**p, **over} if p.get("gdec") and not cfg["window"] and cfg["batch"] * cfg["num_kv_heads"] == 2 else p
submission._plan = plan
CASES = [dict(batch=1, ctx=80000, heads="tp8", layer="full", page=1, dist="coherent"),
         dict(batch=1, ctx=10, heads="tp8", layer="full"),
         dict(batch=1, ctx=1000, heads="tp8", layer="full", page=64, dist="edges"),
         dict(batch=1, ctx=33333, heads="tp8", layer="full", kv_scale=0.015625, dist="soft"),
         dict(batch=1, ctx=2047, heads="tp8", layer="full", page=1, dist="soft"),
         dict(batch=1, ctx=65537, heads="tp8", layer="full", page=64, dist="normal")]
bad = 0
for i, c in enumerate(CASES):
    data = reference.generate_input(seed=4242 + i, **c)
    out = submission.kernel_fn(data); torch.cuda.synchronize()
    ok, msg = reference.check_implementation(data, reference.collect(data, out))
    bad += not ok
    print("PASS" if ok else "FAIL", c, msg[:80], flush=True)
sys.exit(bad)

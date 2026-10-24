"""Decode full cases that reach plan dbal (heads >= 48): correctness against the reference."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import torch, reference, submission  # noqa: E402
CASES = [dict(batch=24, ctx=100, heads="tp8", layer="full", page=1),
         dict(batch=32, ctx=3000, heads="tp8", layer="full", spread="mixed"),
         dict(batch=48, ctx=9000, heads="tp8", layer="full", spread="mixed", dist="edges", page=1),
         dict(batch=32, ctx=20000, heads="tp8", layer="full", kv_scale=0.015625, spread="agentic", dist="soft"),
         dict(batch=24, ctx=8192, heads="dp", layer="full", spread="mixed"),
         dict(batch=40, ctx=2000, heads="dp", layer="full", page=1, dist="coherent"),
         dict(batch=32, ctx=10, heads="tp8", layer="full", spread="mixed"),
         dict(batch=128, ctx=3000, heads="tp8", layer="full", page=1, spread="agentic")]
bad = 0
for i, c in enumerate(CASES):
    data = reference.generate_input(seed=5200 + i, **c)
    plan = submission._plan(data["config"], data["page_table"].shape[1], data["qkv"].device)
    out = submission.kernel_fn(data)
    torch.cuda.synchronize()
    ok, msg = reference.check_implementation(data, reference.collect(data, out))
    bad += not ok
    print(f"{'PASS' if ok else 'FAIL'} dbal={plan.get('dbal', False)} {c} {msg[:60]}", flush=True)
    del data
    torch.cuda.empty_cache()
sys.exit(bad)

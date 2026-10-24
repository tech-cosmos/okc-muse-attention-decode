"""submission.kernel_fn on decode full-layer cases that reach the Gluon decode path (plus the scored ones)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import torch, reference, submission  # noqa: E402
CASES = [dict(batch=16, ctx=80000, heads="tp8", layer="full", page=1, spread="agentic", dist="coherent"),
         dict(batch=32, ctx=80000, heads="tp8", layer="full", page=1, spread="agentic", dist="coherent"),
         dict(batch=8, ctx=8192, heads="tp8", layer="full", spread="mixed"),
         dict(batch=2, ctx=9000, heads="tp8", layer="full", kv_scale=0.015625, page=1),
         dict(batch=4, ctx=80000, heads="tp8", layer="full", spread="mixed", dist="soft"),
         dict(batch=16, ctx=5000, heads="tp8", layer="full", dist="edges", page=1),
         dict(batch=64, ctx=20000, heads="tp8", layer="full", page=1, spread="agentic", dist="soft"),
         dict(batch=3, ctx=100, heads="tp8", layer="full", page=1),
         dict(batch=24, ctx=40000, heads="tp8", layer="full", spread="mixed", dist="edges")]
bad = 0
for i, c in enumerate(CASES):
    data = reference.generate_input(seed=4100 + i, **c)
    plan = submission._plan(data["config"], data["page_table"].shape[1], data["qkv"].device)
    out = submission.kernel_fn(data)
    torch.cuda.synchronize()
    ok, msg = reference.check_implementation(data, reference.collect(data, out))
    bad += not ok
    print(f"{'PASS' if ok else 'FAIL'} gdec={plan.get('gdec', False)} {c} {msg[:70]}", flush=True)
    del data
    torch.cuda.empty_cache()
sys.exit(bad)

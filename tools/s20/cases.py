"""submission.kernel_fn on decode sliding-layer cases with plan overrides (default: the Gluon decode path).
python tools/s20/cases.py ['<plan json>']"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import torch, reference, submission  # noqa: E402
over = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {"gdec": True, "splits": 15, "block_n": 32, "stages": 2,
                                                         "pdt": "bf16"}
orig = submission._plan
submission._plan = lambda cfg, mp, dev: {**orig(cfg, mp, dev), **over} if cfg["window"] and cfg["q_len"] == 1 \
    else orig(cfg, mp, dev)
CASES = [dict(batch=16, ctx=80000, heads="tp8", layer="sliding", page=1, spread="agentic", dist="soft"),
         dict(batch=32, ctx=80000, heads="tp8", layer="sliding", page=1, spread="agentic", dist="soft"),
         dict(batch=1, ctx=80000, heads="tp8", layer="sliding", page=1, dist="soft"),
         dict(batch=1, ctx=10, heads="tp8", layer="sliding"),
         dict(batch=3, ctx=2047, heads="tp8", layer="sliding"),
         dict(batch=4, ctx=6000, heads="tp8", layer="sliding", dist="edges", page=1),
         dict(batch=8, ctx=3000, heads="tp8", layer="sliding", kv_scale=0.015625),
         dict(batch=8, ctx=8192, heads="dp", layer="sliding", spread="mixed"),
         dict(batch=24, ctx=40000, heads="tp8", layer="sliding", spread="mixed", dist="edges"),
         dict(batch=64, ctx=20000, heads="tp8", layer="sliding", page=1, spread="agentic", dist="soft"),
         dict(batch=5, ctx=2049, heads="tp8", layer="sliding", page=1, dist="edges")]
bad = 0
for i, c in enumerate(CASES):
    data = reference.generate_input(seed=4200 + i, **c)
    plan = submission._plan(data["config"], data["page_table"].shape[1], data["qkv"].device)
    out = submission.kernel_fn(data)
    torch.cuda.synchronize()
    ok, msg = reference.check_implementation(data, reference.collect(data, out))
    bad += not ok
    print(f"{'PASS' if ok else 'FAIL'} gdec={plan.get('gdec', False)} {c} {msg[:90]}", flush=True)
    del data
    torch.cuda.empty_cache()
sys.exit(bad)

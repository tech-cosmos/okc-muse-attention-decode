"""agent2: correctness on extra configs for the batched sliding paths. python tools/a2/chk.py"""
import sys, torch
sys.path.insert(0, "/home/daytona/repo2")
import reference, submission
CASES = [
    dict(batch=32, ctx=80000, heads="tp8", layer="sliding", q_len=8, page=1, spread="agentic", dist="soft"),
    dict(batch=32, ctx=6000, heads="tp8", layer="sliding", q_len=8, page=1, dist="edges"),
    dict(batch=32, ctx=3000, heads="tp8", layer="sliding", q_len=8, dist="edges"),
    dict(batch=32, ctx=2047, heads="tp8", layer="sliding", q_len=8, kv_scale=0.015625),
    dict(batch=40, ctx=1500, heads="tp8", layer="sliding", q_len=8, spread="mixed", page=1),
    dict(batch=16, ctx=80000, heads="tp8", layer="sliding", q_len=8, page=1, spread="agentic", dist="soft"),
    dict(batch=16, ctx=6000, heads="tp8", layer="sliding", q_len=8, page=1, dist="edges"),
    dict(batch=16, ctx=80000, heads="tp8", layer="sliding", page=1, spread="agentic", dist="soft"),
    dict(batch=32, ctx=80000, heads="tp8", layer="sliding", page=1, spread="agentic", dist="soft"),
    dict(batch=32, ctx=6000, heads="tp8", layer="sliding", page=1, dist="edges"),
    dict(batch=32, ctx=10, heads="tp8", layer="sliding", page=64),
    dict(batch=24, ctx=2500, heads="tp8", layer="sliding", spread="mixed", kv_scale=0.015625),
]
bad = 0
for i, c in enumerate(CASES):
    data = reference.generate_input(seed=777 + i, **c)
    out = submission.kernel_fn(data); torch.cuda.synchronize()
    ok, msg = reference.check_implementation(data, reference.collect(data, out))
    bad += not ok
    print("PASS" if ok else "FAIL", c, msg[:100], flush=True)
sys.exit(bad)

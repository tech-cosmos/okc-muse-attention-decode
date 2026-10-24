"""Correctness of verify full on gbal shapes (batch >= 16): reference check + error vs the Gluon path."""
import sys, os, torch
sys.path.insert(0, ".")
import reference, submission
cases = [dict(batch=32, ctx=80000, heads="tp8", layer="full", page=1, q_len=8, dist="coherent", spread="agentic"),
         dict(batch=16, ctx=80000, heads="tp8", layer="full", page=1, q_len=8, dist="coherent", spread="agentic"),
         dict(batch=16, ctx=9000, heads="tp8", layer="full", q_len=8, dist="edges", spread="mixed"),
         dict(batch=24, ctx=5000, heads="tp8", layer="full", q_len=8, dist="soft", kv_scale=0.015625, page=1, spread="mixed"),
         dict(batch=17, ctx=3000, heads="tp8", layer="full", q_len=8, dist="coherent", spread="mixed"),
         dict(batch=40, ctx=20000, heads="tp8", layer="full", q_len=8, dist="soft", page=1, spread="agentic")]
for i, c in enumerate(cases):
    data = reference.generate_input(seed=77 + i, **c)
    out = submission.kernel_fn(data)
    torch.cuda.synchronize()
    ok, msg = reference.check_implementation(data, reference.collect(data, out))
    print("PASS" if ok else "FAIL", c, msg[:150], flush=True)

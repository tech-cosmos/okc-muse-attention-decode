"""agent2: correctness of the nvs dispatch (verify sliding, batch >= 32, page 1) on varied inputs vs the reference."""
import sys, torch
sys.path.insert(0, "/home/daytona/repo2")
import reference, submission
cases = [dict(batch=32, ctx=80000, heads="tp8", layer="sliding", page=1, q_len=8, dist="soft", spread="agentic"),
         dict(batch=32, ctx=1500, heads="tp8", layer="sliding", page=1, q_len=8, dist="edges", spread="mixed"),
         dict(batch=40, ctx=5000, heads="tp8", layer="sliding", page=1, q_len=8, dist="soft", kv_scale=0.015625, spread="mixed"),
         dict(batch=33, ctx=3000, heads="tp8", layer="sliding", page=1, q_len=8, dist="coherent", spread="mixed"),
         dict(batch=64, ctx=20000, heads="tp8", layer="sliding", page=1, q_len=8, dist="edges", spread="agentic")]
for i, c in enumerate(cases):
    data = reference.generate_input(seed=177 + i, **c)
    out = submission.kernel_fn(data)
    torch.cuda.synchronize()
    ok, msg = reference.check_implementation(data, reference.collect(data, out))
    print("PASS" if ok else "FAIL", c, msg[:120], flush=True)

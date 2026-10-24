"""Session 44: scases.py plus batch >= 32 verify sliding cases.
Sliding (and Triton full) cases with plan overrides on submission.py: scases.py '<json override>' [file.py]"""
import importlib.util, json, sys
sys.path.insert(0, "/home/daytona/repo")
import torch, reference
path = sys.argv[2] if len(sys.argv) > 2 else "/home/daytona/repo/submission.py"
sp = importlib.util.spec_from_file_location("subx", path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
ov = json.loads(sys.argv[1])
orig = m._plan
m._plan = lambda cfg, mp, dev: {**orig(cfg, mp, dev), **ov}
CASES = [dict(batch=16, ctx=80000, heads="tp8", layer="sliding", page=1, spread="agentic", dist="soft"),
         dict(batch=32, ctx=80000, heads="tp8", layer="sliding", page=1, spread="agentic", dist="soft"),
         dict(batch=16, ctx=80000, heads="tp8", layer="sliding", page=1, q_len=8, spread="agentic", dist="soft"),
         dict(batch=1, ctx=10, heads="tp8", layer="sliding"),
         dict(batch=3, ctx=2047, heads="tp8", layer="sliding"),
         dict(batch=5, ctx=5000, heads="tp8", layer="sliding", q_len=8, spread="mixed", page=1),
         dict(batch=4, ctx=6000, heads="tp8", layer="sliding", dist="edges", page=1),
         dict(batch=3, ctx=4000, heads="tp8", layer="sliding", q_len=8, page=1, kv_scale=0.015625),
         dict(batch=24, ctx=6000, heads="tp8", layer="sliding", dist="edges", page=1, spread="mixed"),
         dict(batch=40, ctx=3000, heads="tp8", layer="sliding", q_len=8, dist="edges", spread="mixed"),
         dict(batch=8, ctx=8192, heads="dp", layer="sliding", spread="mixed"),
         dict(batch=32, ctx=80000, heads="tp8", layer="sliding", page=1, q_len=8, spread="agentic", dist="soft"),
         dict(batch=8, ctx=8192, heads="dp", layer="sliding", q_len=8, spread="mixed"),
         dict(batch=33, ctx=2100, heads="tp8", layer="sliding", q_len=8, dist="edges", spread="mixed", page=1),
         dict(batch=32, ctx=3000, heads="tp8", layer="sliding", q_len=8, kv_scale=0.015625),
         dict(batch=64, ctx=500, heads="tp8", layer="sliding", q_len=8, spread="mixed")]
bad = 0
for i, c in enumerate(CASES):
    data = reference.generate_input(seed=4300 + i, **c)
    out = m.kernel_fn(data)
    torch.cuda.synchronize()
    ok, msg = reference.check_implementation(data, reference.collect(data, out))
    bad += not ok
    print(f"{'PASS' if ok else 'FAIL'} {c} {msg[:90]}", flush=True)
    del data
    torch.cuda.empty_cache()
sys.exit(bad)

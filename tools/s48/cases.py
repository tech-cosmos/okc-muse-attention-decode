"""Session 48: decode cases that hit the heads == 32 sliding plan. Sliding (and Triton full) cases with plan overrides on submission.py: scases.py '<json override>' [file.py]"""
import importlib.util, json, sys
sys.path.insert(0, "/home/daytona/repo")
import torch, reference
path = sys.argv[2] if len(sys.argv) > 2 else "/home/daytona/repo/submission.py"
sp = importlib.util.spec_from_file_location("subx", path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
ov = json.loads(sys.argv[1])
orig = m._plan
m._plan = lambda cfg, mp, dev: {**orig(cfg, mp, dev), **ov}
CASES = [dict(batch=16, ctx=6000, heads="tp8", layer="sliding", dist="edges", spread="mixed"),
         dict(batch=16, ctx=3000, heads="tp8", layer="sliding", page=1, kv_scale=0.015625, spread="mixed"),
         dict(batch=16, ctx=2047, heads="tp8", layer="sliding", page=1),
         dict(batch=2, ctx=8192, heads="dp", layer="sliding", spread="mixed"),
         dict(batch=4, ctx=8192, heads="dp", layer="sliding", page=1),
         dict(batch=16, ctx=80000, heads="tp8", layer="sliding", page=1, spread="agentic", dist="edges")]
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

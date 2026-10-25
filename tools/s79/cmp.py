"""LZ=0 vs LZ=1 outputs (same inputs) on verify-full gbal shapes, plus reference check of LZ=1."""
import sys, os, importlib.util, torch
sys.path.insert(0, "."); sys.path.insert(0, "tools")
import reference
def load(lz, name):
    os.environ["LZ"] = str(lz)
    sp = importlib.util.spec_from_file_location(name, sys.argv[1] if len(sys.argv) > 1 else "submission.py")
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
A, B = load(0, "a"), load(int(os.environ.get("LZB", "1")), "b")
cases = [dict(batch=32, ctx=80000, heads="tp8", layer="full", page=1, q_len=8, dist="coherent", spread="agentic"),
         dict(batch=16, ctx=80000, heads="tp8", layer="full", page=1, q_len=8, dist="coherent", spread="agentic"),
         dict(batch=16, ctx=9000, heads="tp8", layer="full", q_len=8, dist="edges", spread="mixed"),
         dict(batch=24, ctx=5000, heads="tp8", layer="full", q_len=8, dist="soft", kv_scale=0.015625, page=1, spread="mixed"),
         dict(batch=17, ctx=3000, heads="tp8", layer="full", q_len=8, dist="coherent", spread="mixed"),
         dict(batch=40, ctx=20000, heads="tp8", layer="full", q_len=8, dist="soft", page=1, spread="agentic")]
for i, c in enumerate(cases):
    for ks in (None, 64.0):
        data = reference.generate_input(seed=77 + i, **c)
        if ks:  # huge scores: forces the fallback many times (k_scale multiplies every score)
            data["k_scale"] = data["k_scale"] * ks
        oa = A.kernel_fn(dict(data)).float(); ob = B.kernel_fn(dict(data)).float()
        torch.cuda.synchronize()
        d = (oa - ob).norm() / oa.norm()
        ok, msg = reference.check_implementation(data, reference.collect(data, ob.to(oa.dtype) if False else B.kernel_fn(dict(data))))
        print(f"case {i} ks {ks}: rel diff A/B {d:.3e} max {(oa-ob).abs().max():.3e} nan {ob.isnan().any().item()} ref {'PASS' if ok else 'FAIL'} {msg[:60]}", flush=True)

"""LZT=0 vs LZT=1 (Triton _attn_split lazy max) outputs on the shapes that use it, plus reference check."""
import sys, os, importlib.util, torch
sys.path.insert(0, "."); sys.path.insert(0, "tools")
import reference
def load(v, name):
    os.environ["LZT"] = str(v)
    sp = importlib.util.spec_from_file_location(name, "submission.py")
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
A, B = load(0, "a"), load(1, "b")
cases = [dict(batch=1, ctx=80000, heads="tp8", layer="full", page=1, q_len=8, dist="coherent"),
         dict(batch=16, ctx=80000, heads="tp8", layer="sliding", page=1, q_len=8, dist="soft", spread="agentic"),
         dict(batch=32, ctx=80000, heads="tp8", layer="sliding", page=1, q_len=8, dist="soft", spread="agentic"),
         dict(batch=1, ctx=80000, heads="tp8", layer="sliding", page=1, q_len=8, dist="soft"),
         dict(batch=5, ctx=5000, heads="tp8", layer="sliding", q_len=8, spread="mixed", page=1),
         dict(batch=2, ctx=100000, heads="tp8", layer="full", q_len=8, page=1),
         dict(batch=1, ctx=1048576, heads="tp8", layer="full", q_len=8)]
for i, c in enumerate(cases):
    for ks in (None, 64.0):
        data = reference.generate_input(seed=7 + i, **c)
        if ks: data["k_scale"] = data["k_scale"] * ks
        oa = A.kernel_fn(dict(data)).float(); ob = B.kernel_fn(dict(data)).float(); torch.cuda.synchronize()
        ok, msg = reference.check_implementation(data, reference.collect(data, B.kernel_fn(dict(data))))
        print(f"case {i} ks {ks}: rel diff {(oa-ob).norm()/oa.norm():.3e} nan {ob.isnan().any().item()} ref {'PASS' if ok else 'FAIL'} {msg[:40]}", flush=True)

import sys; sys.path.insert(0, "tools"); sys.path.insert(0, ".")
import torch, reference, submission
orig = submission._plan
import os
G = dict(gluon=True, block_n=64, stages=3, lag=1, gpeel=True, pw=4, pregs=40, cregs=96, maxnreg=80, pipe=bool(os.environ.get("PIPE")), crs=bool(os.environ.get("CRS")), h16=bool(os.environ.get("H16")))
def forced(cfg, mp, dev):
    p = orig(cfg, mp, dev)
    if p.get("gluon"): p = dict(p, pipe=bool(os.environ.get("PIPE")), crs=bool(os.environ.get("CRS")), h16=bool(os.environ.get("H16")))
    if not cfg["window"] and cfg["q_len"] == 8 and cfg["num_kv_heads"] % 2 == 0 and not p.get("gluon"):
        pairs = cfg["batch"] * cfg["num_kv_heads"] // 2
        p = dict(G, splits=max(2, 264 // pairs))
    return p
submission._plan = forced
cases = [
 dict(batch=2, ctx=100000, heads="tp8", layer="full", q_len=8, page=1),
 dict(batch=4, ctx=6000, heads="tp8", layer="full", q_len=8, dist="edges"),
 dict(batch=1, ctx=1048576, heads="tp8", layer="full", q_len=8),
 dict(batch=2, ctx=20000, heads="dp", layer="full", q_len=8, dist="soft"),
 dict(batch=2, ctx=20000, heads="dp", layer="full", q_len=8, dist="coherent"),
 dict(batch=6, ctx=9000, heads="tp8", layer="full", q_len=8, kv_scale=0.015625),
 dict(batch=8, ctx=3000, heads="tp8", layer="full", q_len=8, spread="mixed"),
 dict(batch=12, ctx=80000, heads="tp8", layer="full", q_len=8, page=1, spread="agentic", dist="coherent"),
 dict(batch=16, ctx=80000, heads="tp8", layer="full", q_len=8, page=1, spread="agentic", dist="coherent"),
 dict(batch=16, ctx=5000, heads="tp8", layer="full", q_len=8, dist="edges", spread="mixed"),
 dict(batch=20, ctx=30000, heads="tp8", layer="full", q_len=8, page=1, spread="mixed"),
 dict(batch=64, ctx=4000, heads="tp8", layer="full", q_len=8, spread="mixed", dist="edges"),
 dict(batch=40, ctx=20000, heads="tp8", layer="full", q_len=8, page=1, spread="agentic", dist="soft"),
]
for c in cases:
    try:
        data = reference.generate_input(seed=11, **c)
        p = submission._plan(data["config"], data["page_table"].shape[1], data["qkv"].device)
        out = submission.kernel_fn(data)
        ok, msg = reference.check_implementation(data, reference.collect(data, out))
    except Exception as e:
        ok, msg, p = False, repr(e)[:200], {}
    print("PASS" if ok else "FAIL", c, "splits", p.get("splits"), p.get("gluon"), "" if ok else msg[:200], flush=True)
    del data; torch.cuda.empty_cache()

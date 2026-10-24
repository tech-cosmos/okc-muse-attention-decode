"""Dump SASS of _gw_verify for verify full at a batch with a plan override: sass.py batch '<plan json>' out.sass"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import reference, submission, torch  # noqa: E402
batch, plan, out = int(sys.argv[1]), json.loads(sys.argv[2]), sys.argv[3]
orig = submission._plan
submission._plan = lambda cfg, mp, dev: {**orig(cfg, mp, dev), **plan}
c = dict(batch=batch, ctx=80000, heads="tp8", layer="full", page=1, q_len=8, dist="coherent", spread="agentic")
data = reference.generate_input(seed=99, **c)
k = submission._gw_verify
k.device_caches.clear() if hasattr(k, "device_caches") else None
submission.kernel_fn(data); torch.cuda.synchronize()
for dev, cache in k.device_caches.items():
    for key, ck in cache[0].items():
        open(out, "w").write(ck.asm["sass"])
        print("wrote", out, ck.n_regs, ck.n_spills)

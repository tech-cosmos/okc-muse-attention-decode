"""Dump the verify-16 (LOAD env) Gluon kernel's PTX for a plan override: python tools/s22/ptx.py '<json>' out.ptx"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import torch, bench, submission  # noqa: E402
P = json.loads(sys.argv[1])
orig = submission._plan
submission._plan = lambda c, m, dv: {**orig(c, m, dv), **P}
f = submission._gw_verify
r0 = f.run
cap = {}
def run(*a, **k):
    kk = r0(*a, **k); cap['k'] = kk; return kk
f.run = run
for l, c in bench.SCORED:
    if l == os.environ.get("LOAD", "verify 16") and c["layer"] == "full":
        submission.kernel_fn(bench.data_for(l, c)); torch.cuda.synchronize()
k = cap['k']
print("regs", k.n_regs, "spills", k.n_spills)
open(sys.argv[2], "w").write(k.asm['ptx'])

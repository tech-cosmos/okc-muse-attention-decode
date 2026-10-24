"""Compile the verify-16-full Gluon kernel with plan overrides argv[1]; write its PTX/SASS-ready asm to argv[2]."""
import sys, json, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import submission, bench, torch
P = json.loads(sys.argv[1]); orig = submission._plan
submission._plan = lambda c, m, d: {**orig(c, m, d), **P}
f = submission._gw_verify; r0 = f.run; cap = {}
def run(*a, **k):
    kk = r0(*a, **k); cap['k'] = kk; return kk
f.run = run
for l, c in bench.SCORED:
    if l == "verify 16" and c["layer"] == "full":
        submission.kernel_fn(bench.data_for(l, c)); torch.cuda.synchronize()
k = cap['k']; print("regs", k.n_regs, "spills", k.n_spills)
open(sys.argv[2], "w").write(k.asm['ptx'])

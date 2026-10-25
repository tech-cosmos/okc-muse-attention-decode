import sys, json, re; sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import submission, bench, torch, subprocess, tempfile, os
orig = submission._plan
P = json.loads(sys.argv[1])
submission._plan = lambda c, m, dv: {**orig(c, m, dv), **P}
f = submission._gw_verify; r0 = f.run; cap = {}
def run(*a, **k):
    kk = r0(*a, **k); cap['k'] = kk; return kk
f.run = run
for l, c in bench.SCORED:
    if l == "verify 16" and c["layer"] == "full":
        submission.kernel_fn(bench.data_for(l, c)); torch.cuda.synchronize()
k = cap['k']; print(P, "regs", k.n_regs, "spills", k.n_spills)
open("/tmp/s83.sass", "w").write(k.asm['sass']); open("/tmp/s83.ptx", "w").write(k.asm['ptx'])

import sys, json, os; d=sys.argv[1]; sys.path.insert(0, d); sys.path.insert(1, "/home/daytona/repo/tools"); sys.path.insert(2, "/home/daytona/repo")
import submission, bench, torch
orig=submission._plan
for P in json.loads(sys.argv[2]):
    submission._plan=lambda c,m,dv:{**orig(c,m,dv),**P}
    f=submission._gw_verify; r0=f.run
    cap={}
    def run(*a, **k):
        kk=r0(*a, **k); cap['k']=kk; return kk
    f.run=run
    for l, c in bench.SCORED:
        if l == "verify 16" and c["layer"] == "full":
            submission.kernel_fn(bench.data_for(l, c)); torch.cuda.synchronize()
    f.run=r0
    k=cap['k']; print(P, "regs", k.n_regs, "spills", k.n_spills)
import re
print(re.findall(r"setmaxnreg[^;]*;", k.asm['ptx']))
print(re.findall(r"\.maxnreg\s+\d+", k.asm['ptx']))

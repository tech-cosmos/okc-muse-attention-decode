import sys, json; sys.path.insert(0,"/home/daytona/repo"); sys.path.insert(1,"/home/daytona/repo/tools")
import submission, bench, torch, subprocess
orig=submission._plan
P=json.loads(sys.argv[1]); out=sys.argv[2]
submission._plan=lambda c,m,dv:{**orig(c,m,dv),**P}
f=submission._gw_verify; r0=f.run; cap={}
def run(*a, **k):
    kk=r0(*a, **k); cap['k']=kk; return kk
f.run=run
for l, c in bench.SCORED:
    if l == "verify 32" and c["layer"] == "full":
        submission.kernel_fn(bench.data_for(l, c)); torch.cuda.synchronize()
open(out+".cubin","wb").write(cap['k'].asm['cubin'])
s=subprocess.run(["cuobjdump","-sass",out+".cubin"],capture_output=True,text=True).stdout
open(out+".sass","w").write(s)
import re
for op in ["STL","LDL","HGMMA","QGMMA","MUFU.EX2","WARPGROUP.DEPBAR","BAR"]:
    print(op, len(re.findall(r"\b"+re.escape(op), s)))

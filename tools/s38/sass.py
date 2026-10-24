"""SASS of _attn_split for a scored shape: sass.py '<shape>' out_prefix ['<json plan>']"""
import sys, json, re, subprocess
sys.path.insert(0, "/home/daytona/repo"); sys.path.insert(1, "/home/daytona/repo/tools")
import submission, bench, torch
shape, out = sys.argv[1], sys.argv[2]
P = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
orig = submission._plan
submission._plan = lambda c, m, dv: {**orig(c, m, dv), **P}
f = submission._attn_split; r0 = f.run; cap = {}
def run(*a, **k):
    kk = r0(*a, **k); cap['k'] = kk; return kk
f.run = run
for l, c in bench.SCORED:
    if f"{l} {c['layer']}" == shape:
        submission.kernel_fn(bench.data_for(l, c)); torch.cuda.synchronize()
open(out + ".cubin", "wb").write(cap['k'].asm['cubin'])
open(out + ".ptx", "w").write(cap['k'].asm['ptx'])
s = subprocess.run(["/usr/local/cuda-13.0/bin/cuobjdump", "-sass", out + ".cubin"], capture_output=True, text=True).stdout
open(out + ".sass", "w").write(s)
print("regs", cap['k'].n_regs, "spills", cap['k'].n_spills, "shared", cap['k'].metadata.shared)
for op in ["STL", "LDL", "HGMMA", "QGMMA", "MUFU.EX2", "WARPGROUP.DEPBAR", "BAR.SYNC", "LDS", "STS", "LDSM", "STSM", "F2FP", "LDGSTS", "SHFL"]:
    print(op, len(re.findall(r"\b" + re.escape(op), s)))

"""SASS of _gw_verify at a verify-full shape: sass.py batch out_prefix ['<json plan>']"""
import sys, json, re, subprocess
sys.path.insert(0, "/home/daytona/repo"); sys.path.insert(1, "/home/daytona/repo/tools")
import submission, reference, torch
batch, out = int(sys.argv[1]), sys.argv[2]
P = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
orig = submission._plan
submission._plan = lambda c, m, dv: {**orig(c, m, dv), **P}
f = submission._gw_verify; r0 = f.run; cap = {}
def run(*a, **k):
    kk = r0(*a, **k); cap['k'] = kk; return kk
f.run = run
c = dict(batch=batch, ctx=80000, heads="tp8", layer="full", page=1, q_len=8, dist="coherent", spread="agentic")
submission.kernel_fn(reference.generate_input(seed=99, **c)); torch.cuda.synchronize()
open(out + ".cubin", "wb").write(cap['k'].asm['cubin'])
open(out + ".ptx", "w").write(cap['k'].asm['ptx'])
s = subprocess.run(["/usr/local/cuda-13.0/bin/cuobjdump", "-sass", out + ".cubin"], capture_output=True, text=True).stdout
open(out + ".sass", "w").write(s)
print("regs", cap['k'].n_regs, "spills", cap['k'].n_spills, "shared", cap['k'].metadata.shared)
for op in ["STL", "LDL", "HGMMA", "QGMMA", "MUFU.EX2", "LDS", "STS", "F2FP", "PRMT", "LOP3", "SHF"]:
    print(op, len(re.findall(r"\b" + re.escape(op), s)))

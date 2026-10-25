"""agent2: dump the SASS/ttgir of _attn_split for one scored shape. python tools/a2/sass.py 'verify 32' sliding out_prefix"""
import sys, torch, subprocess
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench, submission
load, layer, pre = sys.argv[1], sys.argv[2], sys.argv[3]
kname = sys.argv[4] if len(sys.argv) > 4 else "_attn_split"
orig = getattr(submission, kname)
got = []
class W:
    def __getitem__(self, grid):
        f = orig[grid]
        def run(*a, **k):
            r = f(*a, **k); got.append(r); return r
        return run
setattr(submission, kname, W())
for l, c in bench.SCORED:
    if l == load and c["layer"] == layer:
        data = bench.data_for(l, c)
        submission.kernel_fn(data); torch.cuda.synchronize()
k = got[0]
open(pre + ".ttgir", "w").write(k.asm["ttgir"])
open(pre + ".ptx", "w").write(k.asm["ptx"])
open(pre + ".cubin", "wb").write(k.asm["cubin"])
print(subprocess.run(["/usr/local/cuda/bin/cuobjdump", "-sass", pre + ".cubin"], capture_output=True, text=True).stdout, file=open(pre + ".sass", "w"))
print("regs", k.n_regs, "spills", k.n_spills, "shared", k.metadata.shared)

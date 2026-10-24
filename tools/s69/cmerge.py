"""Graph-timed cost of an in-kernel cluster (DSMEM) split merge: python tools/s69/cmerge.py"""
import os, sys
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import torch, bench
from torch.utils.cpp_extension import load_inline
here = os.path.dirname(os.path.abspath(__file__))
mod = load_inline("cmerge_s69", cpp_sources="void run(int, int, torch::Tensor, torch::Tensor, int); void run_sep(int, torch::Tensor, torch::Tensor, torch::Tensor, int);",
                  cuda_sources=open(os.path.join(here, "cmerge.cu")).read(), functions=["run", "run_sep"], extra_cuda_cflags=["-O3"])
inp = torch.randn(1024 * 16 * 66, device="cuda"); out = torch.zeros(1024 * 1024, device="cuda")
e = bench.empty_cost()
# correctness of the merge at nc 4
mod.run(4, int(os.environ.get('MG', '1')), inp, out, 4); torch.cuda.synchronize()
x = inp[:4 * 16 * 66].view(4, 16 * 66)
acc = x[:, :1024].view(4, 16, 64); m = x[:, 1024:1040]; l = x[:, 1040:1056] + 1
mm = m.max(0).values; w = torch.exp2(m - mm)
ref = (w[:, :, None] * acc).sum(0) / (w * l).sum(0)[:, None]
print("merge err", (out[:1024].view(16, 64) - ref).abs().max().item())
for grid in (32, 256):
    row = []
    for nc, mg in ((1, 0), (4, 4), (16, 4)):
        t = bench.time_graph(lambda: mod.run(nc, mg, inp, out, grid), reps=60) - e
        row.append(f"nc{nc} m{mg} {t:5.2f}")
    part = torch.zeros(grid * 1056, device="cuda")
    for nc in (4, 16):
        t = bench.time_graph(lambda: mod.run_sep(nc, inp, out, part, grid), reps=60) - e
        row.append(f"sep{nc} {t:5.2f}")
    print(f"grid {grid}: " + " | ".join(row), flush=True)
for grid in ():
    row = []
    for nc in (1, 4, 8, 16):
        for mg in (0, 1):
            t = bench.time_graph(lambda: mod.run(nc, mg, inp, out, grid), reps=60) - e
            row.append(f"nc{nc}{'M' if mg else ' '} {t:5.2f}")
    print(f"grid {grid}: " + " | ".join(row), flush=True)

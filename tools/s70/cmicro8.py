"""Consumer-only native microbenchmark (tools/s69/cmicro.cu): correctness vs torch, then graph timing per variant.
python tools/s69/cmicro.py [nt] [grid]"""
import os, sys, fcntl
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
import torch
from torch.utils.cpp_extension import load_inline
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "cmicro8.cu")).read()
mod = load_inline("cmicro_s70_v" + os.environ.get("PV8", "0") + "p" + os.environ.get("NPOLY", "0") + "o" + os.environ.get("ONES", "0") + "l" + os.environ.get("LAZY", "0"), cpp_sources="void run(int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, float, int, torch::Tensor);",
                  cuda_sources=src, functions=["run"], extra_cuda_cflags=["-O3", f"-DNPOLY={os.environ.get('NPOLY', '0')}", f"-DONES={os.environ.get('ONES', '0')}", f"-DLAZY={os.environ.get('LAZY', '0')}", f"-DPV8={os.environ.get('PV8', '0')}", "-lineinfo", "-Xptxas=-v"],
                  verbose=False, build_directory=None)
NS = 3
nt = int(sys.argv[1]) if len(sys.argv) > 1 else 152
grid = int(sys.argv[2]) if len(sys.argv) > 2 else 264
torch.manual_seed(0)
dev = "cuda"
K = (torch.randn(NS, 64, 128, device=dev) * 0.5).to(torch.float8_e4m3fn)
V = (torch.randn(NS, 64, 128, device=dev)).to(torch.float8_e4m3fn)
Q = (torch.randn(2, 64, 64, device=dev)).to(torch.float8_e4m3fn)
Vh = V.view(NS, 64, 2, 64).permute(0, 2, 1, 3).contiguous().to(torch.float16)  # [NS][2][BN][D]
def perm(k):
    kk, c = divmod(k, 32); half, c = divmod(c, 16); t, i = divmod(c, 4)
    return 32 * kk + 16 * half + (2 * t + i if i < 2 else 8 + 2 * t + i - 2)
P = torch.tensor([perm(k) for k in range(64)], device=dev)
VT = V.view(NS, 64, 2, 64)[:, P].permute(0, 2, 3, 1).contiguous().view(torch.uint8)  # [NS][2][d][k]
scale = 0.125 * 1.4426950408889634
def ref(n):
    out = []
    for h in range(2):
        idx = [i % NS for i in range(n)]
        Kh = torch.cat([K[i, :, h * 64:(h + 1) * 64].float() for i in idx])
        Vt = torch.cat([Vh[i, h].float() for i in idx])
        s = Q[h].float() @ Kh.T * scale
        m = s.max(1).values
        p = torch.exp2(s - m[:, None])
        out.append(((p @ Vt) / p.sum(1)[:, None], m))
    return out
for variant in (0, 1, 2):
    for n in (1, 2, 5, 8):
        o = torch.zeros(2, 64, 66, device=dev)
        mod.run(variant, K.view(torch.uint8), Vh, Q.view(torch.uint8), o, n, scale, 1, VT)
        torch.cuda.synchronize()
        r = ref(n)
        err = max(((o[h, :, :64] / o[h, :, 65:66] - r[h][0]).norm() / r[h][0].norm()).item() for h in range(2))
        merr = max((o[h, :, 64] - r[h][1]).abs().max().item() for h in range(2))
        print(f"variant {variant} nt {n}: rel err {err:.2e}, m err {merr:.2e}")
o = torch.zeros(2, 64, 66, device=dev)
def time(variant, reps=30):
    g = torch.cuda.CUDAGraph()
    mod.run(variant, K.view(torch.uint8), Vh, Q.view(torch.uint8), o, nt, scale, grid, VT)
    with torch.cuda.graph(g):
        mod.run(variant, K.view(torch.uint8), Vh, Q.view(torch.uint8), o, nt, scale, grid, VT)
    ts = []
    with open("/tmp/okc-gpu.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for _ in range(reps):
            a, b = torch.cuda.Event(True), torch.cuda.Event(True)
            a.record(); g.replay(); b.record(); torch.cuda.synchronize()
            ts.append(a.elapsed_time(b) * 1000)
    ts.sort()
    return ts[len(ts) // 2]
for variant in [int(v) for v in os.environ.get('VARS', '0,1,2').split(',')]:
    print(f"variant {variant}: nt {nt} grid {grid}: {time(variant):.1f} us")

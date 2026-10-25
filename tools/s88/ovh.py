"""Fixed cost of one / two tiny kernels in a graph after the judge-like read flush (bench.time_graph minus empty)."""
import sys; sys.path.insert(0, "tools"); sys.path.insert(0, ".")
import torch, triton, triton.language as tl
import bench

@triton.jit
def k(x, N: tl.constexpr):
    i = tl.program_id(0)
    tl.store(x + i * N + tl.arange(0, N), tl.load(x + i * N + tl.arange(0, N)) + 1)

x = torch.zeros(1 << 22, device="cuda")
emp = bench.empty_cost()
print("empty", emp)
for name, f in [
    ("1 cta", lambda: k[(1,)](x, N=128)),
    ("512 cta", lambda: k[(512,)](x, N=128)),
    ("2 kernels", lambda: (k[(512,)](x, N=128), k[(16,)](x, N=128))),
    ("2 kernels pdl", lambda: (k[(512,)](x, N=128), k[(16,)](x, N=128, launch_pdl=True))),
    ("3 kernels", lambda: (k[(512,)](x, N=128), k[(16,)](x, N=128), k[(16,)](x, N=128))),
]:
    print(name, "%.2f" % (bench.time_graph(f, reps=100) - emp))

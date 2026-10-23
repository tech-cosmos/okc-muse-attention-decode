import sys; sys.path.insert(0,'/home/daytona/repo/tools'); sys.path.insert(0,'/home/daytona/repo')
import torch, triton, triton.language as tl
import bench
@triton.jit
def chain(p, out, N: tl.constexpr):
    i = tl.program_id(0)
    x = i
    for _ in tl.static_range(N):
        x = tl.load(p + x * 4096 + i)
    tl.store(out + i, x)
P = torch.randint(0, 1<<13, (1<<26,), dtype=torch.int32, device="cuda")
out = torch.zeros(1024, dtype=torch.int32, device='cuda')
for grid in (64, 512):
  for n in range(0,5):
    t = bench.time_graph(lambda: chain[(grid,)](P, out, N=n), reps=200) - bench.empty_cost()
    print(grid, n, round(t,2))

"""Graph-timed vmerge.cu: cluster launch floor (mode 0) vs reduce-scatter merge (mode 1), 2 and 32 clusters."""
import sys, statistics, torch
sys.path.insert(0, "/home/daytona/repo/tools")
import bench
from torch.utils.cpp_extension import load_inline
m = load_inline(name="s74_vmerge4", cpp_sources="void vmerge(torch::Tensor, torch::Tensor, int64_t, int64_t);",
                cuda_sources=open("/home/daytona/repo/tools/s74/vmerge.cu").read(), functions=["vmerge"],
                extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"])
seed = torch.rand(1024, device="cuda")
for ncl in (2, 32):
    out = torch.empty(ncl * 64 * 64, dtype=torch.bfloat16, device="cuda")
    for mode in (0, 3, 2, 3, 2):
        ts = [bench.time_graph(lambda: m.vmerge(seed, out, ncl, mode)) - bench.empty_cost() for _ in range(5)]
        print(f"clusters {ncl:3d} mode {mode}: {statistics.median(ts):6.2f} us", flush=True)

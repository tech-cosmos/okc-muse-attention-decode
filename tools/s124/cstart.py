"""s124: CTA start-time spread (%globaltimer at entry, relative to the earliest CTA) for csl/vtk-shaped launches:
128 threads, optional 16-CTA clusters, each CTA spins SPIN ns (stand-in for the kernel's work) so CTAs co-reside."""
import sys, statistics, torch
from torch.utils.cpp_extension import load_inline
src = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
__global__ void probe(unsigned long long* t, int* sm, long spin) {
    unsigned long long a; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(a));
    int s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
    if (threadIdx.x == 0) { t[blockIdx.x] = a; sm[blockIdx.x] = s; }
    unsigned long long b = a;
    while (b - a < (unsigned long long)spin) asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(b));
}
void run(torch::Tensor t, torch::Tensor sm, int64_t n, int64_t cl, int64_t spin) {
    cudaLaunchConfig_t c = {}; c.gridDim = dim3(n); c.blockDim = dim3(128); c.stream = c10::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at[1]; int na = 0;
    if (cl > 1) { at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = cl; at[0].val.clusterDim.y = 1;
                  at[0].val.clusterDim.z = 1; na = 1;
                  cudaFuncSetAttribute(probe, cudaFuncAttributeNonPortableClusterSizeAllowed, 1); }
    c.attrs = at; c.numAttrs = na;
    cudaLaunchKernelEx(&c, probe, (unsigned long long*)t.data_ptr(), sm.data_ptr<int>(), (long)spin);
}
"""
m = load_inline("cstart124", cpp_sources="void run(torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);",
                cuda_sources=src, functions=["run"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
for n, cl in [(32, 16), (32, 1), (256, 16), (256, 8), (256, 1), (128, 16), (64, 16)]:
    for spin in (0, 3000):
        t = torch.zeros(n, dtype=torch.int64, device="cuda"); sm = torch.zeros(n, dtype=torch.int32, device="cuda")
        p50s, maxs = [], []
        for r in range(30):
            m.run(t, sm, n, cl, spin); torch.cuda.synchronize()
            d = (t - t.min()).tolist(); d.sort()
            if r >= 5: p50s.append(d[n // 2]); maxs.append(d[-1])
        print(f"n {n:3d} cluster {cl:2d} spin {spin:4d}: start spread p50 {statistics.median(p50s):6.0f} ns, "
              f"max {statistics.median(maxs):6.0f} ns, SMs {len(set(sm.tolist()))}")

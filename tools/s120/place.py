"""s120: where does the HW put csl's 16-CTA clusters? Probe kernel with csl's block (128 thr), ~21 KB static smem,
cluster 16, grid 32*B; optional dynamic smem DYN to force fewer CTAs/SM. Prints CTAs per SM per cluster."""
import sys, collections, torch
from torch.utils.cpp_extension import load_inline
src = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
__global__ void __launch_bounds__(128) probe(int* o) {
    __shared__ float pad[5400];
    extern __shared__ float dyn[];
    pad[threadIdx.x] = threadIdx.x; __syncthreads();
    unsigned s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s));
    if (threadIdx.x == 0) o[blockIdx.x] = s + (int)pad[0];
}
void run(torch::Tensor o, int64_t grid, int64_t dyn) {
    cudaLaunchConfig_t cfg = {}; cfg.gridDim = dim3(grid); cfg.blockDim = dim3(128); cfg.dynamicSmemBytes = dyn;
    cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 16; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1; cfg.attrs = at; cfg.numAttrs = 1;
    cudaFuncSetAttribute(probe, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaFuncSetAttribute(probe, cudaFuncAttributeMaxDynamicSharedMemorySize, 200000);
    TORCH_CHECK(cudaLaunchKernelEx(&cfg, probe, o.data_ptr<int>()) == cudaSuccess);
}'''
m = load_inline("s120place", cpp_sources="void run(torch::Tensor, int64_t, int64_t);", cuda_sources=src,
                functions=["run"], extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a"])
for B in (1, 16):
    for dyn in (0, 60000, 120000):
        o = torch.zeros(32 * B, dtype=torch.int32, device="cuda")
        hist = collections.Counter()
        for rep in range(20):
            m.run(o, 32 * B, dyn); torch.cuda.synchronize()
            per = collections.Counter(o.tolist())
            hist[max(per.values())] += 1
            if rep == 0:
                cl = [sorted(collections.Counter(o[i*16:(i+1)*16].tolist()).values(), reverse=True) for i in range(min(2*B, 4))]
                print(f"B{B} dyn {dyn}: SMs used {len(per)}, clusters' CTAs/SM {cl}")
        print(f"   max CTAs on one SM over 20 runs: {dict(hist)}")

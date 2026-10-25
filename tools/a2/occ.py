import torch
from torch.utils.cpp_extension import load_inline
src = r'''
#include <torch/extension.h>
__global__ void __launch_bounds__(384, 1) dummy(int* x) { extern __shared__ int s[]; if (x) x[0] = s[threadIdx.x]; }
std::vector<int64_t> occ(int64_t smem, int64_t cl, int64_t threads) {
  cudaFuncSetAttribute(dummy, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
  cudaFuncSetAttribute(dummy, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  cudaLaunchConfig_t cfg = {}; cfg.gridDim = dim3(128); cfg.blockDim = dim3(threads); cfg.dynamicSmemBytes = smem;
  cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = cl; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1; cfg.attrs = at; cfg.numAttrs = 1;
  int n = -1; cudaError_t e = cudaOccupancyMaxActiveClusters(&n, dummy, &cfg);
  return {n, (int64_t)e};
}
'''
m = load_inline("a2occ", cpp_sources="std::vector<int64_t> occ(int64_t, int64_t, int64_t);", cuda_sources=src, functions=["occ"],
                extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a"])
for smem in (125*1024, 160*1024, 200*1024):
    for cl in (1, 2, 4, 8):
        print(smem//1024, "KB cluster", cl, "max active clusters", m.occ(smem, cl, 384))

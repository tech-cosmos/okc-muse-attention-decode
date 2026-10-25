"""s100: launch floor of a 1-kernel graph vs kernel config (smem, cluster, grid, params), bench.py timing."""
import sys, torch
sys.path.insert(0, "/home/daytona/repo/tools")
import bench
from torch.utils.cpp_extension import load_inline
src = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
struct Big { long long a[64]; };
__global__ void k_empty(float* p) { if (p == nullptr) p[0] = 1; }
__global__ void k_store(float* p) { if (threadIdx.x == 0 && blockIdx.x == 0) p[0] = 1; }
__global__ void k_smem(float* p) { extern __shared__ float s[]; if (p == nullptr) { s[threadIdx.x] = 1; p[0] = s[threadIdx.x ^ 1]; } }
__global__ void k_big(float* p, Big b) { if (p == nullptr) p[0] = b.a[threadIdx.x & 63]; }
__global__ void k_load(float* p, const float* q) { if (threadIdx.x == 0) p[blockIdx.x] = q[blockIdx.x * 64] + 1; }
__global__ void k_chain(float* p, const int* pt, const float* q) { int i = pt[blockIdx.x * 128 + threadIdx.x]; float v = q[(long long)i * 16]; p[blockIdx.x * 128 + threadIdx.x] = v + 1; }
void run(torch::Tensor t, torch::Tensor q, int64_t mode, int64_t grid, int64_t threads, int64_t smem, int64_t cluster) {
  float* p = t.data_ptr<float>();
  auto st = at::cuda::getCurrentCUDAStream();
  cudaLaunchConfig_t cfg = {}; cfg.gridDim = dim3(grid); cfg.blockDim = dim3(threads); cfg.dynamicSmemBytes = smem; cfg.stream = st;
  cudaLaunchAttribute at[1]; int na = 0;
  if (cluster > 1) { at[0].id = cudaLaunchAttributeClusterDimension; at[0].val.clusterDim.x = cluster; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1; na = 1; }
  cfg.attrs = at; cfg.numAttrs = na;
  if (mode == 0) cudaLaunchKernelEx(&cfg, k_empty, p);
  else if (mode == 1) cudaLaunchKernelEx(&cfg, k_store, p);
  else if (mode == 2) { cudaFuncSetAttribute(k_smem, cudaFuncAttributeMaxDynamicSharedMemorySize, 200000); cudaLaunchKernelEx(&cfg, k_smem, p); }
  else if (mode == 3) { Big b = {}; cudaLaunchKernelEx(&cfg, k_big, p, b); }
  else if (mode == 5) cudaLaunchKernelEx(&cfg, k_chain, p, (const int*)(q.data_ptr<float>() + (1 << 21)), (const float*)q.data_ptr<float>());
  else if (mode == 4) cudaLaunchKernelEx(&cfg, k_load, p, (const float*)q.data_ptr<float>());
}
'''
m = load_inline(name="s100_floor", cpp_sources="void run(torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t);",
                cuda_sources="#include <ATen/cuda/CUDAContext.h>\n" + src, functions=["run"],
                extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"])
t = torch.zeros(4096, device="cuda"); q = torch.zeros(1 << 23, device="cuda"); q.view(torch.int32)[1 << 21:(1 << 21) + 32768] = torch.randperm(1 << 17, device="cuda", dtype=torch.int32)[:32768]
cfgs = [("empty 32x128", 0, 32, 128, 0, 1), ("load 32x128", 4, 32, 128, 0, 1), ("chain 32x128", 5, 32, 128, 0, 1), ("chain 256x128", 5, 256, 128, 0, 1)]
e = bench.empty_cost()
print(f"empty graph {e:.2f}")
for r in range(2):
    for name, *a in cfgs:
        print(f"{name:24s} {bench.time_graph(lambda: m.run(t, q, *a)) - e:.2f}", flush=True)

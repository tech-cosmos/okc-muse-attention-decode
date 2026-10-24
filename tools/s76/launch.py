"""What a kernel launch costs under graph replay after a cold-L2 (read) flush, H100: empty graph vs kernels
that do nothing / write via a param / do one dependent load, and grid/smem/cluster variants."""
import sys, torch
sys.path.insert(0, "tools/s76")
from torch.utils.cpp_extension import load_inline
from empty import tg, flushes
src = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
__global__ void k_none() {}
__global__ void k_store(float* p) { if (threadIdx.x == 0 && blockIdx.x == 0) p[0] = 1.f; }
__global__ void k_load(const float* q, float* p) { if (threadIdx.x == 0 && blockIdx.x == 0) p[0] = q[0] + 1.f; }
__global__ void k_chain(const int* q, float* p) { if (threadIdx.x == 0 && blockIdx.x == 0) { int a = q[0]; a = q[a]; p[0] = q[a]; } }
__global__ void __cluster_dims__(8,1,1) k_clu(float* p) { if (threadIdx.x == 0 && blockIdx.x == 0) p[0] = 1.f; }
__global__ void k_pdl_first(float* p) { asm volatile("griddepcontrol.launch_dependents;"); if (threadIdx.x == 0 && blockIdx.x == 0) p[0] = 1.f; }
__global__ void k_pdl_second(float* p) { asm volatile("griddepcontrol.wait;" ::: "memory"); if (threadIdx.x == 0 && blockIdx.x == 0) p[1] = p[0] + 1.f; }
void pdl2(torch::Tensor p, int g, int t, int use) {
  auto st = at::cuda::getCurrentCUDAStream();
  k_pdl_first<<<g, t, 0, st>>>(p.data_ptr<float>());
  cudaLaunchConfig_t cfg = {}; cfg.gridDim = g; cfg.blockDim = t; cfg.stream = st;
  cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = use;
  cfg.attrs = at; cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, k_pdl_second, p.data_ptr<float>());
}
void none(int g, int t) { k_none<<<g, t, 0, at::cuda::getCurrentCUDAStream()>>>(); }
void store(torch::Tensor p, int g, int t) { k_store<<<g, t, 0, at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<float>()); }
void load(torch::Tensor q, torch::Tensor p, int g, int t) { k_load<<<g, t, 0, at::cuda::getCurrentCUDAStream()>>>(q.data_ptr<float>(), p.data_ptr<float>()); }
void chain(torch::Tensor q, torch::Tensor p, int g, int t) { k_chain<<<g, t, 0, at::cuda::getCurrentCUDAStream()>>>(q.data_ptr<int>(), p.data_ptr<float>()); }
void clu(torch::Tensor p, int g, int t) { k_clu<<<g, t, 0, at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<float>()); }
'''
m = load_inline("s76launch", cpp_sources="void none(int,int); void store(torch::Tensor,int,int); void load(torch::Tensor,torch::Tensor,int,int); void chain(torch::Tensor,torch::Tensor,int,int); void clu(torch::Tensor,int,int); void pdl2(torch::Tensor,int,int,int);",
                cuda_sources=src.replace("#include <torch/extension.h>", "#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>"),
                functions=["none", "store", "load", "chain", "clu", "pdl2"], extra_cuda_cflags=["-O3", "-arch=sm_90a"], verbose=False)
p = torch.zeros(1, device="cuda"); q = torch.zeros(1 << 20, device="cuda")
qi = torch.zeros(1 << 20, dtype=torch.int32, device="cuda"); qi[0] = 4096 * 7; qi[4096 * 7] = 4096 * 50
fl = flushes["read"]
e0 = tg(lambda: None, fl)
print(f"empty graph {e0:.2f}")
for name, fn in [("none 1x32", lambda: m.none(1, 32)), ("none 132x256", lambda: m.none(132, 256)),
                 ("none 2048x256", lambda: m.none(2048, 256)), ("none 2048x32", lambda: m.none(2048, 32)),
                 ("none 1024x64", lambda: m.none(1024, 64)), ("none 512x128", lambda: m.none(512, 128)), ("none 256x64", lambda: m.none(256, 64)),
                 ("store 1x32", lambda: m.store(p, 1, 32)), ("load 1x32", lambda: m.load(q, p, 1, 32)),
                 ("chain3 1x32", lambda: m.chain(qi, p, 1, 32)), ("cluster8 store 16x32", lambda: m.clu(p, 16, 32)),
                 ("2x none", lambda: (m.none(1, 32), m.none(1, 32))), ("2x store", lambda: (m.store(p, 1, 32), m.store(p, 1, 32))),
                 ("torch add_", lambda: p.add_(1)), ("2 kernels no pdl", lambda: m.pdl2(q, 1, 32, 0)), ("2 kernels pdl", lambda: m.pdl2(q, 1, 32, 1)),
                 ("2 kernels pdl 132x128", lambda: m.pdl2(q, 132, 128, 1))]:
    print(f"{name:24s} {tg(fn, fl) - e0:6.2f} us over empty")
print("no flush:")
nf = lambda: None
e0n = tg(lambda: None, nf)
for name, fn in [("none 1x32", lambda: m.none(1, 32)), ("load 1x32", lambda: m.load(q, p, 1, 32)), ("2x none", lambda: (m.none(1, 32), m.none(1, 32)))]:
    print(f"{name:24s} {tg(fn, nf) - e0n:6.2f} us over empty (empty {e0n:.2f})")

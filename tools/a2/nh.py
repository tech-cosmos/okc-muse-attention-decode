"""s8: gather rate for a no-merge verify-sliding grid: one CTA per (request, KV head) reads the whole window's 64-B
K and V halves of head h into a D-deep ring of 8-KB tiles (64 keys) with cp.async (T threads), page-table slice staged
in smem in chunks. Question: can a smem-feeding gather on 2B CTAs reach nm's register rate (12.1 / 9.1 us)?
python tools/a2/nh.py"""
import sys, torch
sys.path.insert(0, "/home/daytona/repo2/tools"); sys.path.insert(0, "/home/daytona/repo2")
import bench
from torch.utils.cpp_extension import load_inline
src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
template<int T, int D, int HS>
__global__ void __launch_bounds__(T, 1) h2(const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc, const int* __restrict__ pt,
    const int* __restrict__ seq, int ptw, unsigned* out) {
  constexpr int RB = 128 / HS, TB = 64 * RB * 2;        // row bytes per head slice, tile bytes (K + V)
  extern __shared__ __align__(1024) uint8_t sm[];
  int* spt = (int*)(sm + D * TB);                       // 2048 ints
  const int b = blockIdx.x / HS, h = blockIdx.x % HS, tid = threadIdx.x;
  const int s = seq[b], lo = max(0, s - 2047), n = (s - lo + 63) / 64;
  for (int t = tid; t < s - lo; t += T) spt[t] = __ldg(pt + (long)b * ptw + lo + t);
  __syncthreads();
  uint4 acc = make_uint4(0, 0, 0, 0);
  constexpr int CPR = RB / 16, PER = 64 * CPR * 2 / T;  // 16-B chunks per row; cp16 per thread per tile
  for (int i = 0; i < n + D; ++i) {
    if (i >= D) {
      asm volatile("cp.async.wait_group %0;" ::"n"(D - 1) : "memory"); __syncthreads();
      const uint4 x = reinterpret_cast<const uint4*>(sm + ((i - D) % D) * TB)[tid % (TB / 16)];
      acc.x ^= x.x; acc.y ^= x.y; acc.z ^= x.z; acc.w ^= x.w;
      __syncthreads();
    }
    if (i < n) {
      uint8_t* dst = sm + (i % D) * TB;
      const int lim = s - lo - 1 - i * 64;
#pragma unroll
      for (int q = 0; q < PER; ++q) {
        const int u = tid + T * q, isv = u / (64 * CPR), r = (u / CPR) % 64, c = u % CPR;
        const long off = (long)spt[i * 64 + min(r, lim)] * 128 + h * RB + c * 16;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(sa(dst + isv * 64 * RB + r * RB + ((c ^ (r & (CPR - 1))) << 4))), "l"((isv ? vc : kc) + off) : "memory");
      }
      asm volatile("cp.async.commit_group;" ::: "memory");
    }
  }
  unsigned x = acc.x ^ acc.y ^ acc.z ^ acc.w;
  if (x == 0x12345678u) out[0] = x;
}
template<int T, int D, int HS> void L(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seq, torch::Tensor out) {
  const int smem = D * 64 * (128 / HS) * 2 + 8192;
  cudaFuncSetAttribute(h2<T, D, HS>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  const int B = seq.size(0);
  h2<T, D, HS><<<HS * B, T, smem, at::cuda::getCurrentCUDAStream()>>>((const uint8_t*)kc.data_ptr(), (const uint8_t*)vc.data_ptr(),
     pt.data_ptr<int>(), seq.data_ptr<int>(), pt.size(1), (unsigned*)out.data_ptr());
}
void run(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seq, torch::Tensor out, int64_t v) {
  switch (v) {
    case 0: L<128, 4, 2>(kc, vc, pt, seq, out); break;
    case 1: L<128, 8, 2>(kc, vc, pt, seq, out); break;
    case 2: L<256, 4, 2>(kc, vc, pt, seq, out); break;
    case 3: L<256, 8, 2>(kc, vc, pt, seq, out); break;
    case 4: L<256, 12, 2>(kc, vc, pt, seq, out); break;
    case 5: L<512, 4, 2>(kc, vc, pt, seq, out); break;
    case 6: L<512, 8, 2>(kc, vc, pt, seq, out); break;
    case 7: L<256, 4, 1>(kc, vc, pt, seq, out); break;
    case 8: L<512, 4, 1>(kc, vc, pt, seq, out); break;
    case 9: L<512, 6, 1>(kc, vc, pt, seq, out); break;
  }
}
"""
m = load_inline(name="a2_nh1", cpp_sources="void run(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t);",
                cuda_sources=src, functions=["run"], extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"])
names = ["HS2 T128 D4", "HS2 T128 D8", "HS2 T256 D4", "HS2 T256 D8", "HS2 T256 D12", "HS2 T512 D4", "HS2 T512 D8", "HS1 T256 D4", "HS1 T512 D4", "HS1 T512 D6"]
vs = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else list(range(len(names)))
out = torch.zeros(4, dtype=torch.int32, device="cuda")
for load, c in bench.SCORED:
    if c["layer"] != "sliding" or c["batch"] == 1 or c["q_len"] != 8: continue
    data = bench.data_for(load, c)
    kc, vc, pt, seq = data["k_cache"], data["v_cache"], data["page_table"].int(), data["seq_lens"].int()
    e = bench.empty_cost()
    res = [bench.time_graph(lambda: m.run(kc, vc, pt, seq, out, v)) - e for v in vs]
    print(f"{load}: " + "  ".join(f"{names[v]} {x:.2f}" for v, x in zip(vs, res)), flush=True)

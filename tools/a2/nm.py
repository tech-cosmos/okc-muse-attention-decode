"""s7: can a no-split (no-merge) batched-sliding kernel gather fast enough? Gather each request's 2048-token window
with only HS CTAs per request (HS 1: one CTA reads both heads' 128-B rows; HS 2: one CTA per KV head reads 64-B
halves; HS 4: two CTAs per KV head read the same halves (L2 reuse)). Registers-held loads, R tokens per thread
per iteration, page table prefetched one iteration ahead.  python tools/a2/nm.py"""
import sys, torch
sys.path.insert(0, "/home/daytona/repo2/tools"); sys.path.insert(0, "/home/daytona/repo2")
import bench
from torch.utils.cpp_extension import load_inline
src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
template<int HS, int R, int T>
__global__ void __launch_bounds__(T, 1) nmg(const uint4* __restrict__ kc, const uint4* __restrict__ vc, const int* __restrict__ pt,
    const int* __restrict__ seq, int ptw, unsigned* out) {
  constexpr int HB = HS == 1 ? 128 : 64, TPT = HB / 16, TOK = T / TPT * R;   // tokens per iteration
  const int b = blockIdx.y, head = HS == 1 ? 0 : (blockIdx.x & 1);
  const int s = seq[b], lo = max(0, s - 2048);
  const int grp = threadIdx.x / TPT, lane = threadIdx.x % TPT;
  const int off = head * 4 + lane;   // 16-B chunk index inside the 128-B row
  uint4 acc = make_uint4(0, 0, 0, 0);
  int idx[R], nidx[R];
  #pragma unroll
  for (int r = 0; r < R; r++) { int t = lo + grp * R + r; idx[r] = t < s ? pt[(long)b * ptw + t] : -1; }
  for (int base = lo; base < s; base += TOK) {
    #pragma unroll
    for (int r = 0; r < R; r++) { int t = base + TOK + grp * R + r; nidx[r] = t < s ? __ldg(pt + (long)b * ptw + t) : -1; }
    uint4 k[R], v[R];
    #pragma unroll
    for (int r = 0; r < R; r++) {
      if (idx[r] >= 0) { k[r] = __ldg(kc + (long)idx[r] * 8 + off); v[r] = __ldg(vc + (long)idx[r] * 8 + off); }
      else { k[r] = make_uint4(0, 0, 0, 0); v[r] = k[r]; }
    }
    #pragma unroll
    for (int r = 0; r < R; r++) { acc.x ^= k[r].x ^ v[r].x; acc.y ^= k[r].y ^ v[r].y; acc.z ^= k[r].z ^ v[r].z; acc.w ^= k[r].w ^ v[r].w; }
    #pragma unroll
    for (int r = 0; r < R; r++) idx[r] = nidx[r];
  }
  unsigned x = acc.x ^ acc.y ^ acc.z ^ acc.w;
  if (x == 0x12345678u) out[0] = x;
}
template<int SPL, int R, int T>
__global__ void __launch_bounds__(T, 1) nms(const uint4* __restrict__ kc, const uint4* __restrict__ vc, const int* __restrict__ pt,
    const int* __restrict__ seq, int ptw, unsigned* out) {
  // split version: CTA (split, b) gathers keys [lo + split * 2048 / SPL, +2048 / SPL), 128-B rows, registers.
  constexpr int TPT = 8, TOK = T / TPT * R, CH = 2048 / SPL;
  const int b = blockIdx.y;
  const int s = seq[b], lo = max(0, s - 2048) + blockIdx.x * CH, hi = min(lo + CH, s);
  const int grp = threadIdx.x / TPT, lane = threadIdx.x % TPT;
  uint4 acc = make_uint4(0, 0, 0, 0);
  for (int base = lo; base < hi; base += TOK) {
    int idx[R];
    #pragma unroll
    for (int r = 0; r < R; r++) { int t = base + grp * R + r; idx[r] = t < hi ? __ldg(pt + (long)b * ptw + t) : -1; }
    uint4 k[R], v[R];
    #pragma unroll
    for (int r = 0; r < R; r++) {
      if (idx[r] >= 0) { k[r] = __ldg(kc + (long)idx[r] * 8 + lane); v[r] = __ldg(vc + (long)idx[r] * 8 + lane); }
      else { k[r] = make_uint4(0, 0, 0, 0); v[r] = k[r]; }
    }
    #pragma unroll
    for (int r = 0; r < R; r++) { acc.x ^= k[r].x ^ v[r].x; acc.y ^= k[r].y ^ v[r].y; acc.z ^= k[r].z ^ v[r].z; acc.w ^= k[r].w ^ v[r].w; }
  }
  unsigned x = acc.x ^ acc.y ^ acc.z ^ acc.w;
  if (x == 0x12345678u) out[0] = x;
}
template<int SPL, int R, int T> void launchs(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seq, torch::Tensor out) {
  dim3 grid(SPL, seq.size(0));
  nms<SPL, R, T><<<grid, T, 0, at::cuda::getCurrentCUDAStream()>>>((const uint4*)kc.data_ptr(), (const uint4*)vc.data_ptr(), pt.data_ptr<int>(), seq.data_ptr<int>(), pt.size(1), (unsigned*)out.data_ptr());
}
#define VS(n, SPL, R, T) case n: launchs<SPL, R, T>(kc, vc, pt, seq, out); break;
template<int R, int T, int DEP>
__global__ void __launch_bounds__(T, 1) nmc(const uint4* __restrict__ kc, const uint4* __restrict__ vc, const int* __restrict__ pt,
    const int* __restrict__ seq, int ptw, unsigned* out) {
  // cp.async version of HS 1: per iteration R tokens x (K, V) x 16 B per thread into a DEP-deep per-thread smem ring.
  extern __shared__ uint4 ring[];
  __shared__ int spt[2048];
  constexpr int TPT = 8, TOK = T / TPT * R;
  const int b = blockIdx.y;
  const int s = seq[b], lo = max(0, s - 2048);
  const int grp = threadIdx.x / TPT, lane = threadIdx.x % TPT;
  uint4 acc = make_uint4(0, 0, 0, 0);
  int idx[R];
  const int nit = (s - lo + TOK - 1) / TOK;
  for (int t = threadIdx.x; t < s - lo; t += T) spt[t] = __ldg(pt + (long)b * ptw + lo + t);
  __syncthreads();
  auto issue = [&](int it) {
    #pragma unroll
    for (int r = 0; r < R; r++) {
      int t = lo + it * TOK + grp * R + r;
      if (t < s) {
        int id = spt[t - lo];
        unsigned d = (unsigned)__cvta_generic_to_shared(&ring[((it % DEP) * R * 2 + 2 * r) * T + threadIdx.x]);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(d), "l"(kc + (long)id * 8 + lane) : "memory");
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(d + T * 16), "l"(vc + (long)id * 8 + lane) : "memory");
      }
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  };
  for (int it = 0; it < DEP - 1; ++it) issue(it);
  for (int it = 0; it < nit; ++it) {
    issue(it + DEP - 1);
    asm volatile("cp.async.wait_group %0;" ::"n"(DEP - 1) : "memory");
    #pragma unroll
    for (int r = 0; r < 2 * R; r++) { uint4 x = ring[((it % DEP) * R * 2 + r) * T + threadIdx.x]; acc.x ^= x.x; acc.y ^= x.y; acc.z ^= x.z; acc.w ^= x.w; }
  }
  unsigned x = acc.x ^ acc.y ^ acc.z ^ acc.w;
  if (x == 0x12345678u) out[0] = x;
}
template<int R, int T, int DEP> void launchc(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seq, torch::Tensor out) {
  dim3 grid(1, seq.size(0));
  const int smem = DEP * R * 2 * T * 16;
  cudaFuncSetAttribute(nmc<R, T, DEP>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  nmc<R, T, DEP><<<grid, T, smem, at::cuda::getCurrentCUDAStream()>>>((const uint4*)kc.data_ptr(), (const uint4*)vc.data_ptr(), pt.data_ptr<int>(), seq.data_ptr<int>(), pt.size(1), (unsigned*)out.data_ptr());
}
#define VC(n, R, T, DEP) case n: launchc<R, T, DEP>(kc, vc, pt, seq, out); break;
template<int HS, int R, int T> void launch(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seq, torch::Tensor out) {
  dim3 grid(HS, seq.size(0));
  nmg<HS, R, T><<<grid, T, 0, at::cuda::getCurrentCUDAStream()>>>((const uint4*)kc.data_ptr(), (const uint4*)vc.data_ptr(), pt.data_ptr<int>(), seq.data_ptr<int>(), pt.size(1), (unsigned*)out.data_ptr());
}
#define V(n, HS, R, T) case n: launch<HS, R, T>(kc, vc, pt, seq, out); break;
void run(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seq, torch::Tensor out, int64_t v) {
  switch (v) {
    V(0, 1, 4, 512) V(1, 1, 8, 512) V(2, 1, 4, 1024) V(3, 1, 8, 1024)
    V(4, 2, 4, 512) V(5, 2, 8, 512) V(6, 2, 4, 1024) V(7, 2, 8, 1024) V(8, 2, 16, 512)
    V(9, 4, 4, 512) V(10, 4, 8, 512) V(11, 4, 8, 1024) V(12, 4, 16, 512)
    VC(13, 2, 512, 4) VC(14, 4, 512, 3) VC(15, 2, 1024, 3) VC(16, 1, 1024, 6) VC(17, 4, 256, 6) VC(18, 2, 256, 12) VC(19, 4, 128, 12) VC(20, 8, 128, 6) VC(21, 8, 256, 3)
    VS(22, 4, 8, 512) VS(23, 4, 4, 1024) VS(24, 4, 8, 256) VS(25, 8, 8, 256) VS(26, 8, 4, 512) VS(27, 2, 8, 512) VS(28, 16, 4, 256)
  }
}
"""
m = load_inline(name="a2_nm4", cpp_sources="void run(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t);",
                cuda_sources=src, functions=["run"], extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"])
names = ["HS1 R4 T512", "HS1 R8 T512", "HS1 R4 T1024", "HS1 R8 T1024", "HS2 R4 T512", "HS2 R8 T512", "HS2 R4 T1024",
         "HS2 R8 T1024", "HS2 R16 T512", "HS4 R4 T512", "HS4 R8 T512", "HS4 R8 T1024", "HS4 R16 T512", "cp R2 T512 D4", "cp R4 T512 D3", "cp R2 T1024 D3", "cp R1 T1024 D6", "cp R4 T256 D6", "cp R2 T256 D12", "cp R4 T128 D12", "cp R8 T128 D6", "cp R8 T256 D3", "S4 R8 T512", "S4 R4 T1024", "S4 R8 T256", "S8 R8 T256", "S8 R4 T512", "S2 R8 T512", "S16 R4 T256"]
out = torch.zeros(4, dtype=torch.int32, device="cuda")
for load, c in bench.SCORED:
    if c["layer"] != "sliding" or c["batch"] == 1: continue
    data = bench.data_for(load, c)
    kc, vc, pt, seq = data["k_cache"], data["v_cache"], data["page_table"].int(), data["seq_lens"].int()
    e = bench.empty_cost()
    res = [bench.time_graph(lambda: m.run(kc, vc, pt, seq, out, v)) - e for v in [3, 22, 23, 24, 25, 26, 27, 28]]
    print(f"{load} q{c['q_len']}: " + "  ".join(f"{n} {x:.2f}" for n, x in zip([names[i] for i in [3, 22, 23, 24, 25, 26, 27, 28]], res)), flush=True)

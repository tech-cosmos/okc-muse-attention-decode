"""s8: gather mechanism on nvs's grid (S splits x B CTAs, 64-key tiles of both K and V 128-B rows = 16 KB/tile).
Page-table slice staged in smem once, then a D-deep ring of 16-KB slots. Modes:
 0 cp16 (nvs mapping: 8 x 16-B cp.async per thread per tile, 128 thr), wait_group + bar.sync per tile
 1 cp.async.bulk 128-B rows (1 per thread per tile, 128 thr), mbarrier complete_tx
 2 cp16 with 512 threads (2 per thread per tile)
 3 cp16 + cp.async.mbarrier.arrive.noinc completion (no wait_group / bar.sync)
 4 bulk with 64 threads issuing 2 rows each
python tools/a2/nmb.py"""
import sys, torch
sys.path.insert(0, "/home/daytona/repo2/tools"); sys.path.insert(0, "/home/daytona/repo2")
import bench
from torch.utils.cpp_extension import load_inline
src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>
__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbw(uint32_t b, uint32_t par) {
  asm volatile("{\n.reg .pred p;\nW: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n" ::"r"(b), "r"(par) : "memory");
}
template<int MODE, int T, int D, int X = 0, int ST = 0>
__global__ void __launch_bounds__(T + X, 1) g(const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc, const int* __restrict__ pt,
    const int* __restrict__ seq, int ptw, int S, int CH, unsigned* out, float4* __restrict__ pbuf) {
  extern __shared__ __align__(1024) uint8_t sm[];
  uint8_t* ring = sm;                                   // D x 16 KB
  int* spt = (int*)(sm + D * 16384);                    // CH ints
  uint64_t* mb = (uint64_t*)(sm + D * 16384 + 4 * 1024);
  const int b = blockIdx.x / S, split = blockIdx.x % S, tid = threadIdx.x;
  if (X && tid >= T) { if (seq[0] == -5) out[1] = 1; return; }
  const int s = seq[b], lo = max(0, s - 2048) + split * CH, hi = min(lo + CH, s);
  const int n = hi > lo ? (hi - lo + 63) / 64 : 0;
  if (tid == 0) for (int i = 0; i < D; ++i) asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(sa(&mb[i])), "r"(MODE == 3 ? T : 1) : "memory");
  for (int t = tid; t < hi - lo; t += T) spt[t] = __ldg(pt + (long)b * ptw + lo + t);
  asm volatile("bar.sync 1, %0;" ::"n"(T) : "memory");
  uint4 acc = make_uint4(0, 0, 0, 0);
  if (ST == 2) for (int j = 0; j < 2048 / T; ++j) pbuf[blockIdx.x * 2048 + j * T + tid] = make_float4(seq[0], 1, 2, 3);
  for (int i = 0; i < n + D; ++i) {
    if (i >= D) {
      const int sl = (i - D) % D; const uint32_t par = ((i - D) / D) & 1;
      if (MODE == 0 || MODE == 2) { asm volatile("cp.async.wait_group %0;" ::"n"(D - 1) : "memory"); asm volatile("bar.sync 1, %0;" ::"n"(T) : "memory"); }
      else mbw(sa(&mb[sl]), par);
      uint4 x = reinterpret_cast<const uint4*>(ring + sl * 16384)[tid];
      acc.x ^= x.x; acc.y ^= x.y; acc.z ^= x.z; acc.w ^= x.w;
      asm volatile("bar.sync 1, %0;" ::"n"(T) : "memory");
    }
    if (i < n) {
      const int sl = i % D, kb = i * 64;
      uint8_t* dst = ring + sl * 16384;
      const int lim = hi - lo - 1 - kb;
      if (MODE == 0 || MODE == 3 || MODE == 2) {
        constexpr int RPT = T / 8;  // rows per pass
        const int grow = tid >> 3, ch = tid & 7;
#pragma unroll
        for (int j = 0; j < 64 / RPT; ++j) {
          const int r = grow + RPT * j; const long off = (long)spt[kb + min(r, lim)] * 128 + ch * 16;
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(sa(dst + r * 128 + ((ch ^ (r & 7)) << 4))), "l"(kc + off) : "memory");
          asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(sa(dst + 8192 + r * 128 + ((ch ^ (r & 7)) << 4))), "l"(vc + off) : "memory");
        }
        if (MODE == 3) asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(sa(&mb[sl])) : "memory");
        else asm volatile("cp.async.commit_group;" ::: "memory");
      } else {
        if (tid == 0) asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(sa(&mb[sl])), "r"(16384) : "memory");
        constexpr int PER = 128 / T;
#pragma unroll
        for (int q = 0; q < PER; ++q) {
          const int u = tid + T * q, r = u & 63, isv = u >> 6;
          const long off = (long)spt[kb + min(r, lim)] * 128;
          asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 128, [%2];"
                       ::"r"(sa(dst + isv * 8192 + r * 128)), "l"((isv ? vc : kc) + off), "r"(sa(&mb[sl])) : "memory");
        }
      }
    }
  }
  if (ST == 1) for (int j = 0; j < 2048 / T; ++j) pbuf[blockIdx.x * 2048 + j * T + tid] = make_float4(acc.x, acc.y, 2, 3);
  unsigned x = acc.x ^ acc.y ^ acc.z ^ acc.w;
  if (x == 0x12345678u) out[0] = x;
}
static torch::Tensor PBUF;
void setpb(torch::Tensor b) { PBUF = b; }
template<int MODE, int T, int D, int X = 0, int PAD = 0, int ST = 0> void L(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seq, int S, torch::Tensor out) {
  const int smem = D * 16384 + 4096 + 64 + PAD;
  cudaFuncSetAttribute(g<MODE, T, D, X, ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  const int B = seq.size(0);
  g<MODE, T, D, X, ST><<<S * B, T + X, smem, at::cuda::getCurrentCUDAStream()>>>((const uint8_t*)kc.data_ptr(), (const uint8_t*)vc.data_ptr(),
     pt.data_ptr<int>(), seq.data_ptr<int>(), pt.size(1), S, 2048 / S, (unsigned*)out.data_ptr(), (float4*)PBUF.data_ptr());
}
void run(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seq, int64_t S, torch::Tensor out, int64_t v) {
  switch (v) {
    case 0: L<0, 128, 2>(kc, vc, pt, seq, S, out); break;
    case 1: L<0, 128, 4>(kc, vc, pt, seq, S, out); break;
    case 2: L<0, 128, 8>(kc, vc, pt, seq, S, out); break;
    case 3: L<1, 128, 2>(kc, vc, pt, seq, S, out); break;
    case 4: L<1, 128, 4>(kc, vc, pt, seq, S, out); break;
    case 5: L<1, 128, 8>(kc, vc, pt, seq, S, out); break;
    case 6: L<2, 512, 4>(kc, vc, pt, seq, S, out); break;
    case 7: L<2, 512, 8>(kc, vc, pt, seq, S, out); break;
    case 8: L<3, 128, 4>(kc, vc, pt, seq, S, out); break;
    case 9: L<3, 128, 8>(kc, vc, pt, seq, S, out); break;
    case 10: L<1, 64, 8>(kc, vc, pt, seq, S, out); break;
    case 11: L<0, 128, 2, 256>(kc, vc, pt, seq, S, out); break;
    case 12: L<0, 128, 2, 0, 90000>(kc, vc, pt, seq, S, out); break;
    case 13: L<0, 128, 2, 256, 90000>(kc, vc, pt, seq, S, out); break;
    case 14: L<0, 128, 2, 0, 0, 1>(kc, vc, pt, seq, S, out); break;
    case 15: L<0, 128, 2, 0, 0, 2>(kc, vc, pt, seq, S, out); break;
  }
}
"""
m = load_inline(name="a2_nmb4", cpp_sources="void setpb(torch::Tensor); void run(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, torch::Tensor, int64_t);",
                cuda_sources=src, functions=["run", "setpb"], extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"])
names = ["cp16 D2", "cp16 D4", "cp16 D8", "bulk D2", "bulk D4", "bulk D8", "cp16x512 D4", "cp16x512 D8", "cpmb D4", "cpmb D8", "bulk64 D8", "cp16 D2 +256thr", "cp16 D2 +88KB", "cp16 D2 +both", "cp16 D2 st-end", "cp16 D2 st-start"]
vs = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else list(range(len(names)))
out = torch.zeros(4, dtype=torch.int32, device="cuda")
pbuf = torch.empty(512 * 2048 * 4, dtype=torch.float32, device="cuda"); m.setpb(pbuf)
for load, c in bench.SCORED:
    if c["layer"] != "sliding" or c["batch"] == 1: continue
    data = bench.data_for(load, c)
    kc, vc, pt, seq = data["k_cache"], data["v_cache"], data["page_table"].int(), data["seq_lens"].int()
    e = bench.empty_cost()
    for S in ([4] if c["batch"] == 32 else [8, 4]):
        res = [bench.time_graph(lambda: m.run(kc, vc, pt, seq, S, out, v)) - e for v in vs]
        print(f"{load} q{c['q_len']} S{S}: " + "  ".join(f"{names[v]} {x:.2f}" for v, x in zip(vs, res)), flush=True)

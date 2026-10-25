# agent2 s22: tools/s85/sgather.py + 64-B half-row variants (10 z-split, 14 interleaved, 16-18 L2::128B/256B/64B hints).
"""s85: gather floor for sliding shapes (B x 2048 random 128-B K and V rows, with the seq -> page table -> row chain).
Native kernel, each thread owns R rows (K and V, 8 x 16-B loads each), all loads issued before any use.
  python tools/s85/sgather.py"""
import sys, torch
sys.path.insert(0, "/home/daytona/repo2/tools"); sys.path.insert(0, "/home/daytona/repo2")
import bench, submission
from torch.utils.cpp_extension import load_inline
src = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
template<int R, int TPR, int SPEC>
__global__ void __launch_bounds__(256) sg(const uint4* __restrict__ kc, const uint4* __restrict__ vc, const int* __restrict__ pt,
    const int* __restrict__ seq, int ptw, int win, unsigned* out) {
  // block handles rows of one request; TPR threads per 128-B row (each 128/TPR bytes); R rows per thread group
  int b = blockIdx.y;
  int s = SPEC ? ptw - 1 : seq[b];
  if (SPEC == 2) { int s2 = seq[b]; if (s2 != s) s = s2; }
  int lo = max(0, s - win);
  int per = 16 / TPR * 0 + (128 / 16) / TPR;   // 16-B chunks per thread per row
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int grp = tid / TPR, lane = tid % TPR;
  uint4 acc = make_uint4(0,0,0,0);
  int idx[R];
  #pragma unroll
  for (int r = 0; r < R; r++) { int t = lo + grp * R + r; idx[r] = t < s ? pt[(long)b * ptw + t] : -1; }
  uint4 kv[R][2][128/16/TPR];
  #pragma unroll
  for (int r = 0; r < R; r++) {
    if (idx[r] >= 0) {
      #pragma unroll
      for (int c = 0; c < 128/16/TPR; c++) { kv[r][0][c] = __ldg(kc + (long)idx[r] * 8 + lane * (8/TPR) + c); kv[r][1][c] = __ldg(vc + (long)idx[r] * 8 + lane * (8/TPR) + c); }
    } else {
      #pragma unroll
      for (int c = 0; c < 128/16/TPR; c++) { kv[r][0][c] = make_uint4(0,0,0,0); kv[r][1][c] = make_uint4(0,0,0,0); }
    }
  }
  #pragma unroll
  for (int r = 0; r < R; r++)
    #pragma unroll
    for (int c = 0; c < 128/16/TPR; c++) { acc.x ^= kv[r][0][c].x ^ kv[r][1][c].x; acc.y ^= kv[r][0][c].y ^ kv[r][1][c].y; acc.z ^= kv[r][0][c].z ^ kv[r][1][c].z; acc.w ^= kv[r][0][c].w ^ kv[r][1][c].w; }
  unsigned v = acc.x ^ acc.y ^ acc.z ^ acc.w;
  if (v == 0x12345678u) out[0] = v;
}

template<int P> __device__ __forceinline__ uint4 ldp(const uint4* p) {
  uint4 r;
  if (P == 128) asm volatile("ld.global.nc.L2::128B.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  else if (P == 256) asm volatile("ld.global.nc.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  else if (P == 64) asm volatile("ld.global.nc.L2::64B.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  else r = __ldg(p);
  return r;
}
template<int R, int TPR, int SPEC, int HB, int P>
__global__ void __launch_bounds__(256) sgp(const uint4* __restrict__ kc, const uint4* __restrict__ vc, const int* __restrict__ pt,
    const int* __restrict__ seq, int ptw, int win, unsigned* out) {
  // block handles rows of one request; TPR threads per 128-B row (each 128/TPR bytes); R rows per thread group
  int b = blockIdx.y;
  int s = SPEC ? ptw - 1 : seq[b];
  if (SPEC == 2) { int s2 = seq[b]; if (s2 != s) s = s2; }
  int lo = max(0, s - win);
     // 16-B chunks per thread per row
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int grp = tid / TPR, lane = tid % TPR;
  uint4 acc = make_uint4(0,0,0,0);
  int idx[R];
  #pragma unroll
  for (int r = 0; r < R; r++) { int t = lo + grp * R + r; idx[r] = t < s ? pt[(long)b * ptw + t] : -1; }
  uint4 kv[R][2][HB/16/TPR];
  #pragma unroll
  for (int r = 0; r < R; r++) {
    if (idx[r] >= 0) {
      #pragma unroll
      for (int c = 0; c < HB/16/TPR; c++) { kv[r][0][c] = ldp<P>(kc + (long)idx[r] * 8 + blockIdx.z * (HB/16) + lane * ((HB/16)/TPR) + c); kv[r][1][c] = ldp<P>(vc + (long)idx[r] * 8 + blockIdx.z * (HB/16) + lane * ((HB/16)/TPR) + c); }
    } else {
      #pragma unroll
      for (int c = 0; c < HB/16/TPR; c++) { kv[r][0][c] = make_uint4(0,0,0,0); kv[r][1][c] = make_uint4(0,0,0,0); }
    }
  }
  #pragma unroll
  for (int r = 0; r < R; r++)
    #pragma unroll
    for (int c = 0; c < HB/16/TPR; c++) { acc.x ^= kv[r][0][c].x ^ kv[r][1][c].x; acc.y ^= kv[r][0][c].y ^ kv[r][1][c].y; acc.z ^= kv[r][0][c].z ^ kv[r][1][c].z; acc.w ^= kv[r][0][c].w ^ kv[r][1][c].w; }
  unsigned v = acc.x ^ acc.y ^ acc.z ^ acc.w;
  if (v == 0x12345678u) out[0] = v;
}

template<int R, int TPR, int SPEC, int HB>
__global__ void __launch_bounds__(256) sgh(const uint4* __restrict__ kc, const uint4* __restrict__ vc, const int* __restrict__ pt,
    const int* __restrict__ seq, int ptw, int win, unsigned* out) {
  // block handles rows of one request; TPR threads per 128-B row (each 128/TPR bytes); R rows per thread group
  int b = blockIdx.y;
  int s = SPEC ? ptw - 1 : seq[b];
  if (SPEC == 2) { int s2 = seq[b]; if (s2 != s) s = s2; }
  int lo = max(0, s - win);
     // 16-B chunks per thread per row
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int grp = tid / TPR, lane = tid % TPR;
  uint4 acc = make_uint4(0,0,0,0);
  int idx[R];
  #pragma unroll
  for (int r = 0; r < R; r++) { int t = lo + grp * R + r; idx[r] = t < s ? pt[(long)b * ptw + t] : -1; }
  uint4 kv[R][2][HB/16/TPR];
  #pragma unroll
  for (int r = 0; r < R; r++) {
    if (idx[r] >= 0) {
      #pragma unroll
      for (int c = 0; c < HB/16/TPR; c++) { kv[r][0][c] = __ldg(kc + (long)idx[r] * 8 + blockIdx.z * (HB/16) + lane * ((HB/16)/TPR) + c); kv[r][1][c] = __ldg(vc + (long)idx[r] * 8 + blockIdx.z * (HB/16) + lane * ((HB/16)/TPR) + c); }
    } else {
      #pragma unroll
      for (int c = 0; c < HB/16/TPR; c++) { kv[r][0][c] = make_uint4(0,0,0,0); kv[r][1][c] = make_uint4(0,0,0,0); }
    }
  }
  #pragma unroll
  for (int r = 0; r < R; r++)
    #pragma unroll
    for (int c = 0; c < HB/16/TPR; c++) { acc.x ^= kv[r][0][c].x ^ kv[r][1][c].x; acc.y ^= kv[r][0][c].y ^ kv[r][1][c].y; acc.z ^= kv[r][0][c].z ^ kv[r][1][c].z; acc.w ^= kv[r][0][c].w ^ kv[r][1][c].w; }
  unsigned v = acc.x ^ acc.y ^ acc.z ^ acc.w;
  if (v == 0x12345678u) out[0] = v;
}

template<int R, int TPR, int SPEC, int HB>
__global__ void __launch_bounds__(256) sgi(const uint4* __restrict__ kc, const uint4* __restrict__ vc, const int* __restrict__ pt,
    const int* __restrict__ seq, int ptw, int win, unsigned* out) {
  // block handles rows of one request; TPR threads per 128-B row (each 128/TPR bytes); R rows per thread group
  int b = blockIdx.y;
  int s = SPEC ? ptw - 1 : seq[b];
  if (SPEC == 2) { int s2 = seq[b]; if (s2 != s) s = s2; }
  int lo = max(0, s - win);
     // 16-B chunks per thread per row
  int tid = (blockIdx.x >> 1) * blockDim.x + threadIdx.x;
  int grp = tid / TPR, lane = tid % TPR;
  uint4 acc = make_uint4(0,0,0,0);
  int idx[R];
  #pragma unroll
  for (int r = 0; r < R; r++) { int t = lo + grp * R + r; idx[r] = t < s ? pt[(long)b * ptw + t] : -1; }
  uint4 kv[R][2][HB/16/TPR];
  #pragma unroll
  for (int r = 0; r < R; r++) {
    if (idx[r] >= 0) {
      #pragma unroll
      for (int c = 0; c < HB/16/TPR; c++) { kv[r][0][c] = __ldg(kc + (long)idx[r] * 8 + (blockIdx.x & 1) * (HB/16) + lane * ((HB/16)/TPR) + c); kv[r][1][c] = __ldg(vc + (long)idx[r] * 8 + (blockIdx.x & 1) * (HB/16) + lane * ((HB/16)/TPR) + c); }
    } else {
      #pragma unroll
      for (int c = 0; c < HB/16/TPR; c++) { kv[r][0][c] = make_uint4(0,0,0,0); kv[r][1][c] = make_uint4(0,0,0,0); }
    }
  }
  #pragma unroll
  for (int r = 0; r < R; r++)
    #pragma unroll
    for (int c = 0; c < HB/16/TPR; c++) { acc.x ^= kv[r][0][c].x ^ kv[r][1][c].x; acc.y ^= kv[r][0][c].y ^ kv[r][1][c].y; acc.z ^= kv[r][0][c].z ^ kv[r][1][c].z; acc.w ^= kv[r][0][c].w ^ kv[r][1][c].w; }
  unsigned v = acc.x ^ acc.y ^ acc.z ^ acc.w;
  if (v == 0x12345678u) out[0] = v;
}


template<int R, int TPR, int HB> void launchi(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seq, torch::Tensor out, int win) {
  int B = seq.size(0);
  int threads = win / R * TPR;
  dim3 grid(2 * ((threads + 255) / 256), B, 1);
  sgi<R, TPR, 0, HB><<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>((const uint4*)kc.data_ptr(), (const uint4*)vc.data_ptr(), pt.data_ptr<int>(), seq.data_ptr<int>(), pt.size(1), win, (unsigned*)out.data_ptr());
}
template<int R, int TPR, int HB, int P> void launchp(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seq, torch::Tensor out, int win) {
  int B = seq.size(0);
  int threads = win / R * TPR;
  dim3 grid((threads + 255) / 256, B, 128 / HB);
  sgp<R, TPR, 0, HB, P><<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>((const uint4*)kc.data_ptr(), (const uint4*)vc.data_ptr(), pt.data_ptr<int>(), seq.data_ptr<int>(), pt.size(1), win, (unsigned*)out.data_ptr());
}
template<int R, int TPR, int HB> void launchh(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seq, torch::Tensor out, int win) {
  int B = seq.size(0);
  int threads = win / R * TPR;
  dim3 grid((threads + 255) / 256, B, 128 / HB);
  sgh<R, TPR, 0, HB><<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>((const uint4*)kc.data_ptr(), (const uint4*)vc.data_ptr(), pt.data_ptr<int>(), seq.data_ptr<int>(), pt.size(1), win, (unsigned*)out.data_ptr());
}
template<int R, int TPR, int SPEC=0> void launch(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seq, torch::Tensor out, int win) {
  int B = seq.size(0);
  int threads = win / R * TPR;
  dim3 grid((threads + 255) / 256, B);
  sg<R, TPR, SPEC><<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>((const uint4*)kc.data_ptr(), (const uint4*)vc.data_ptr(), pt.data_ptr<int>(), seq.data_ptr<int>(), pt.size(1), win, (unsigned*)out.data_ptr());
}
void run(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seq, torch::Tensor out, int64_t win, int64_t v) {
  switch (v) {
    case 0: launch<1, 8>(kc, vc, pt, seq, out, win); break;
    case 1: launch<2, 8>(kc, vc, pt, seq, out, win); break;
    case 2: launch<4, 8>(kc, vc, pt, seq, out, win); break;
    case 3: launch<1, 4>(kc, vc, pt, seq, out, win); break;
    case 4: launch<2, 4>(kc, vc, pt, seq, out, win); break;
    case 5: launch<4, 4>(kc, vc, pt, seq, out, win); break;
    case 6: launch<8, 8>(kc, vc, pt, seq, out, win); break;
    case 7: launch<2, 2>(kc, vc, pt, seq, out, win); break;
    case 8: launch<1, 8, 1>(kc, vc, pt, seq, out, win); break;
    case 9: launch<1, 8, 2>(kc, vc, pt, seq, out, win); break;
    case 10: launchh<1, 4, 64>(kc, vc, pt, seq, out, win); break;
    case 11: launchh<2, 4, 64>(kc, vc, pt, seq, out, win); break;
    case 12: launchh<1, 8, 128>(kc, vc, pt, seq, out, win); break;
    case 14: launchi<1, 4, 64>(kc, vc, pt, seq, out, win); break;
    case 15: launchi<2, 4, 64>(kc, vc, pt, seq, out, win); break;
    case 16: launchp<1, 4, 64, 128>(kc, vc, pt, seq, out, win); break;
    case 17: launchp<1, 4, 64, 256>(kc, vc, pt, seq, out, win); break;
    case 18: launchp<1, 4, 64, 64>(kc, vc, pt, seq, out, win); break;
    case 19: launchp<1, 8, 128, 128>(kc, vc, pt, seq, out, win); break;
    case 13: launchh<1, 2, 32>(kc, vc, pt, seq, out, win); break;
  }
}
"""
m = load_inline(name="a2s22_sg3", cpp_sources="void run(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t);",
                cuda_sources=src, functions=["run"], extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"])
out = torch.zeros(4, dtype=torch.int32, device="cuda")
for load, c in bench.SCORED:
    if c["layer"] != "sliding" or c["q_len"] != 1 or c["batch"] == 1: continue
    data = bench.data_for(load, c)
    kc, vc, pt, seq = data["k_cache"], data["v_cache"], data["page_table"], data["seq_lens"]
    print(load, kc.shape, kc.dtype, pt.shape, pt.dtype, seq.dtype, seq[:4].tolist())
    pt = pt.int(); seq = seq.int()
    base = bench.time_graph(lambda: submission.kernel_fn(data)) - bench.empty_cost()
    res = []
    for v in (0, 10, 18, 16, 17, 19):
        res.append(bench.time_graph(lambda: m.run(kc, vc, pt, seq, out, 2048, v)) - bench.empty_cost())
    print(f"{load} sliding: main {base:.2f} us; gather R/TPR variants " + " ".join(f"{x:.2f}" for x in res), flush=True)

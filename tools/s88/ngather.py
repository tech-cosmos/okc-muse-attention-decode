"""Native batch-1 full gather floor (H100): every 16-B chunk of the 80000 K and V rows loaded with max loads in flight
(page-table entries first, then all row chunks per thread unrolled), bench.time_graph timing."""
import sys; sys.path.insert(0, "tools"); sys.path.insert(0, ".")
import torch
from torch.utils.cpp_extension import load_inline
import bench
src = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
template <int R>
__global__ void __launch_bounds__(256) g(const uint4* kc, const uint4* vc, const int* pt, int n, int dense, uint4* out) {
    // thread t handles chunk c = t % 8 of rows (t / 8) * R .. + R (contiguous rows per thread group)
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = t & 7, r0 = (t >> 3) * R;
    int rows[R];
#pragma unroll
    for (int i = 0; i < R; ++i) { int r = min(r0 + i * 0 + i, n - 1); rows[i] = dense ? r : pt[r]; }
    uint4 a = make_uint4(0, 0, 0, 0);
    uint4 kv[2 * R];
#pragma unroll
    for (int i = 0; i < R; ++i) { kv[2 * i] = kc[(long)rows[i] * 8 + c]; kv[2 * i + 1] = vc[(long)rows[i] * 8 + c]; }
#pragma unroll
    for (int i = 0; i < 2 * R; ++i) { a.x ^= kv[i].x; a.y ^= kv[i].y; a.z ^= kv[i].z; a.w ^= kv[i].w; }
    if (r0 < n && (a.x == 0x12345678)) out[t] = a;
}
void run(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, int64_t n, int64_t r, int64_t dense, int64_t bs, torch::Tensor out) {
    auto st = c10::cuda::getCurrentCUDAStream();
    long th = (n + r - 1) / r * 8;
    int grid = (th + bs - 1) / bs;
#define L(R) if (r == R) g<R><<<grid, bs, 0, st>>>((const uint4*)kc.data_ptr(), (const uint4*)vc.data_ptr(), pt.data_ptr<int>(), n, dense, (uint4*)out.data_ptr());
    L(1) L(2) L(4) L(8) L(16)
}
'''
m = load_inline(name="s88_ng", cpp_sources="void run(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, torch::Tensor);",
                cuda_sources=src, functions=["run"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])
c = dict(batch=1, ctx=80000, heads="tp8", layer="full", page=1, q_len=1, dist="coherent")
d = bench.data_for("decode 1", c)
kc, vc = d["k_cache"].view(torch.uint8), d["v_cache"].view(torch.uint8)
pt = d["page_table"][0].contiguous(); n = int(d["seq_lens"][0])
out = torch.empty(1 << 22, dtype=torch.int32, device="cuda")
emp = bench.empty_cost()
for dense in (0, 1):
    for r in (1, 2, 4, 8, 16):
        for bs in (128, 256):
            print("dense %d R %d bs %d: %.2f us" % (dense, r, bs, bench.time_graph(lambda: m.run(kc, vc, pt, n, r, dense, bs, out)) - emp), flush=True)

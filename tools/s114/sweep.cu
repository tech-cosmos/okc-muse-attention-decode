// s114: gathers-only ceiling of a "windowed L2 sweep" for decode full (no global inversion).
// Each warp = one program owning a list of K/V rows (decode 32 full rows of one request), pre-bucketed by physical
// window (diag only: built outside the timed call). Round r: the warp prefetches its 1/P share of window r + D into
// L2 (cp.async.bulk.prefetch.L2, dense), then gathers its own rows of window r (8 lanes x 16 B per 128-B row,
// K and V). D < 0: no prefetch. With rows in page-table order and one "window" this is the plain random gather.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

__device__ __forceinline__ void pf_l2(const void* p, unsigned bytes) {
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(bytes) : "memory");
}

template <int UNR>
__global__ void __launch_bounds__(512) sweep(const uint4* __restrict__ kc, const uint4* __restrict__ vc,
                                             const int* __restrict__ rows, const int* __restrict__ off, int NWIN,
                                             long win_bytes, long total, int D, int P, int* __restrict__ sink) {
    const int lane = threadIdx.x & 31;
    const int p = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (p >= P) return;
    const int sub = lane & 7, grp = lane >> 3;  // 4 rows per warp step, 8 lanes x 16 B each
    const long share = ((win_bytes / P) + 15) & ~15L;
    const char* kb = reinterpret_cast<const char*>(kc);
    const char* vb = reinterpret_cast<const char*>(vc);
    uint4 acc = make_uint4(0, 0, 0, 0);
    const int* o = off + (long)p * (NWIN + 1);
    bool only = false, line = false;
    if (D >= 300) { only = true; line = true; D -= 300; }
    else if (D >= 200) { line = true; D -= 200; }
    else if (D >= 100) { only = true; D -= 100; }
    if (line && D > 0)
        for (int r = 0; r < D && r < NWIN; ++r) {
            const long a = r * win_bytes + p * share;
            for (long x = a + lane * 128; x < a + share && x < total; x += 32 * 128) {
                asm volatile("prefetch.global.L2 [%0];" ::"l"(kb + x));
                asm volatile("prefetch.global.L2 [%0];" ::"l"(vb + x));
            }
        }
    if (!line && D > 0 && lane == 0)
        for (int r = 0; r < D && r < NWIN; ++r) {
            const long a = r * win_bytes + p * share;
            for (long x = a; x < a + share && x < total; x += 32768) {
                const unsigned n = (unsigned)min(32768L, min(a + share, total) - x);
                pf_l2(kb + x, n);
                pf_l2(vb + x, n);
            }
        }
    for (int r = 0; r < NWIN; ++r) {
        if (line && D > 0 && r + D < NWIN) {
            const long a = (long)(r + D) * win_bytes + p * share;
            for (long x = a + lane * 128; x < a + share && x < total; x += 32 * 128) {
                asm volatile("prefetch.global.L2 [%0];" ::"l"(kb + x));
                asm volatile("prefetch.global.L2 [%0];" ::"l"(vb + x));
            }
        }
        if (!line && D > 0 && lane == 0 && r + D < NWIN) {
            const long a = (long)(r + D) * win_bytes + p * share;
            for (long x = a; x < a + share && x < total; x += 32768) {
                const unsigned n = (unsigned)min(32768L, min(a + share, total) - x);
                pf_l2(kb + x, n);
                pf_l2(vb + x, n);
            }
        }
        if (only) continue;
        const int a = __ldg(o + r), e = __ldg(o + r + 1);
        for (int i = a + grp; i < e; i += 4 * UNR) {
            uint4 k[UNR], v[UNR];
#pragma unroll
            for (int u = 0; u < UNR; ++u) {
                const int j = i + 4 * u;
                if (j < e) {
                    const long row = __ldg(rows + j);
                    k[u] = __ldg(kc + row * 8 + sub);
                    v[u] = __ldg(vc + row * 8 + sub);
                } else {
                    k[u] = make_uint4(0, 0, 0, 0);
                    v[u] = k[u];
                }
            }
#pragma unroll
            for (int u = 0; u < UNR; ++u) {
                acc.x ^= k[u].x ^ v[u].x; acc.y ^= k[u].y ^ v[u].y;
                acc.z ^= k[u].z ^ v[u].z; acc.w ^= k[u].w ^ v[u].w;
            }
        }
    }
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x9e3779b9u) sink[0] = 1;
}

// Dense ceiling: every warp reads a contiguous range of rows (same byte count as the lists).
__global__ void __launch_bounds__(512) dense(const uint4* __restrict__ kc, const uint4* __restrict__ vc, long nrows,
                                             int* __restrict__ sink) {
    const long nvec = nrows * 8;
    uint4 acc = make_uint4(0, 0, 0, 0);
    const long stride = (long)gridDim.x * blockDim.x;
    long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
    for (; i + 3 * stride < nvec; i += 4 * stride) {
        uint4 k[4], v[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) { k[u] = __ldg(kc + i + u * stride); v[u] = __ldg(vc + i + u * stride); }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            acc.x ^= k[u].x ^ v[u].x; acc.y ^= k[u].y ^ v[u].y; acc.z ^= k[u].z ^ v[u].z; acc.w ^= k[u].w ^ v[u].w;
        }
    }
    for (; i < nvec; i += stride) { const uint4 k = __ldg(kc + i), v = __ldg(vc + i); acc.x ^= k.x ^ v.x; }
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x9e3779b9u) sink[0] = 1;
}

void sweep_run(torch::Tensor kc, torch::Tensor vc, torch::Tensor rows, torch::Tensor off, int64_t nwin, int64_t win_bytes,
               int64_t d, int64_t P, int64_t nt, torch::Tensor sink) {
    const long total = kc.numel();
    auto st = c10::cuda::getCurrentCUDAStream();
    const int wpb = (int)nt / 32;
    const int unr = d >= 1000 ? (int)(d / 1000) : 4;
    if (d >= 1000) d = d % 1000 - 1;
#define SW(U) sweep<U><<<(P + wpb - 1) / wpb, nt, 0, st>>>((const uint4*)kc.data_ptr(), (const uint4*)vc.data_ptr(), \
        rows.data_ptr<int>(), off.data_ptr<int>(), (int)nwin, win_bytes, total, (int)d, (int)P, sink.data_ptr<int>())
    if (unr == 8) SW(8); else if (unr == 16) SW(16); else SW(4);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dense_run(torch::Tensor kc, torch::Tensor vc, int64_t nrows, torch::Tensor sink) {
    auto st = c10::cuda::getCurrentCUDAStream();
    dense<<<132 * 4, 512, 0, st>>>((const uint4*)kc.data_ptr(), (const uint4*)vc.data_ptr(), nrows, sink.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

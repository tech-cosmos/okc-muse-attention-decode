SRC = r'''// psort: per-program bucketing of the page table for the decode-full kernel (_gd_full, page 1).
// CTA p covers exactly _gd_full program p's key range [lo, hi) of request b (equal splits or dbal ranges, same
// integer arithmetic) and writes those entries to out[b, lo:hi) grouped by physical row >> SH (64 buckets, order
// inside a bucket arbitrary). Keys inside one program may be visited in any order on full layers (softmax over a
// set), so _gd_full reading `out` instead of the page table computes the same attention; all programs then sweep
// the pool from low to high rows together, which reads DRAM in a more local order.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
namespace ps {
constexpr int NB = 64, NT = 256, CAP = 4096;
__global__ void __launch_bounds__(NT) psort(const int* __restrict__ pt, const int* __restrict__ seqs, int* __restrict__ out,
                                            int B, int MP, int S, int BN, int NP, int dbal, int SH) {
    __shared__ int hist[NB];
    __shared__ int buf[CAP];
    __shared__ int sb, slo, shi;
    const int tid = threadIdx.x, pid = blockIdx.x;
    if (tid < 32) {
        // Program pid's request and range, as in _gd_full_body.
        int b, split, nsp;
        if (dbal) {
            int tl = 0, sq = 0;
            if (tid < B) { sq = seqs[tid]; tl = (sq + BN - 1) / BN; }
            int inc = tl;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) { int v = __shfl_up_sync(0xffffffff, inc, o); if (tid >= o) inc += v; }
            int tt = __shfl_sync(0xffffffff, inc, 31); tt = max(tt, 1);
            int excl = inc - tl;
            int c = tid + (int)((long long)(NP - B) * excl / tt);
            unsigned le = __ballot_sync(0xffffffff, tid < B && c <= pid);
            b = __popc(le) - 1;
            int cb = __shfl_sync(0xffffffff, c, b);
            int cn = __shfl_sync(0xffffffff, c, (b + 1) & 31);
            if (b + 1 >= B) cn = NP;
            split = pid - cb; nsp = cn - cb;
        } else {
            b = pid / S; split = pid % S; nsp = S;
        }
        if (tid == 0) {
            int seq = seqs[b];
            int chunk = ((seq + nsp - 1) / nsp + BN - 1) / BN * BN;
            int lo = split * chunk, hi = min(lo + chunk, seq);
            sb = b; slo = lo; shi = max(hi, lo);
        }
    }
    __syncthreads();
    const int b = sb, lo0 = slo, hi = shi;
    const int* src = pt + (long)b * MP;
    int* dst = out + (long)b * MP;
    for (int lo = lo0; lo < hi; lo += CAP) {
        const int n = min(CAP, hi - lo);
        if (tid < NB) hist[tid] = 0;
        __syncthreads();
        constexpr int PER = CAP / NT;
        int v[PER], r[PER];
#pragma unroll
        for (int i = 0; i < PER; ++i) {
            int k = i * NT + tid;
            v[i] = k < n ? src[lo + k] : 0;
        }
#pragma unroll
        for (int i = 0; i < PER; ++i) {
            int k = i * NT + tid;
            r[i] = k < n ? atomicAdd(&hist[min(v[i] >> SH, NB - 1)], 1) : 0;
        }
        __syncthreads();
        if (tid < 32) {
            int a0 = hist[2 * tid], a1 = hist[2 * tid + 1], s = a0 + a1, inc = s;
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) { int t = __shfl_up_sync(0xffffffff, inc, o); if (tid >= o) inc += t; }
            hist[2 * tid] = inc - s; hist[2 * tid + 1] = inc - s + a0;
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < PER; ++i) {
            int k = i * NT + tid;
            if (k < n) buf[hist[min(v[i] >> SH, NB - 1)] + r[i]] = v[i];
        }
        __syncthreads();
        for (int k = tid; k < n; k += NT) dst[lo + k] = buf[k];
        __syncthreads();
    }
}
}  // namespace ps
void psort(torch::Tensor pt, torch::Tensor seq, torch::Tensor out, int64_t S, int64_t BN, int64_t NP, int64_t dbal,
           int64_t SH) {
    int B = seq.size(0), MP = pt.size(1);
    TORCH_CHECK(B <= 32, "psort: batch <= 32");
    int grid = dbal ? NP : S * B;
    ps::psort<<<grid, ps::NT, 0, c10::cuda::getCurrentCUDAStream()>>>(pt.data_ptr<int>(), seq.data_ptr<int>(),
        out.data_ptr<int>(), B, MP, S, BN, NP, dbal, SH);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''

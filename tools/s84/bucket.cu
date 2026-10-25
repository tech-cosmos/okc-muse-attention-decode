// s84: cheap prototype of the physical-order "multi-split pass" (NOTES s80): bucket every valid page-table entry
// (request b, position t) of a page-1 decode-full step by physical window phys / WS into per-window lists.
// One kernel: CTA = (slice of ENT positions, request b); smem histogram (rank by smem atomic), one global atomic
// per non-empty window to reserve a run, then the entries packed (b << 20 | t) into window w's list at w * CAP.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

template <int ENT, int NW>
__global__ void __launch_bounds__(512) bucket(const int* __restrict__ pt, const int* __restrict__ seqs, int MP, int WS,
                                              int CAP, int* __restrict__ gcount, int* __restrict__ lists, int abl) {
    __shared__ int hist[NW], base[NW], pref[NW], sent[ENT], total;
    __shared__ unsigned short swin[ENT];
    const int b = blockIdx.y, t0 = blockIdx.x * ENT, seq = seqs[b];
    if (t0 >= seq) return;
    if (abl == 3) { int z = 0; for (int k = threadIdx.x; k < ENT; k += 512) z ^= t0 + k < seq ? pt[(long)b * MP + t0 + k] : 0; if (z == 1234567) lists[0] = z; return; }
    for (int i = threadIdx.x; i < NW; i += blockDim.x) hist[i] = 0;
    __syncthreads();
    constexpr int PER = ENT / 512;
    int win[PER], rk[PER];
    const int* row = pt + (long)b * MP;
#pragma unroll
    for (int k = 0; k < PER; ++k) {
        const int t = t0 + k * 512 + threadIdx.x;
        win[k] = -1;
        if (t < seq) {
            win[k] = row[t] / WS;
            rk[k] = atomicAdd(&hist[win[k]], 1);
        }
    }
    __syncthreads();
    // Exclusive scan of hist into pref (warp 0, NW / 32 bins per lane), and reserve each window's run.
    if (threadIdx.x < 32) {
        constexpr int PL = NW / 32;
        int v[PL], sum = 0;
#pragma unroll
        for (int k = 0; k < PL; ++k) { v[k] = hist[threadIdx.x * PL + k]; sum += v[k]; }
        int inc = sum;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) { int y = __shfl_up_sync(0xffffffff, inc, o); if (threadIdx.x >= o) inc += y; }
        int run = inc - sum;
#pragma unroll
        for (int k = 0; k < PL; ++k) {
            const int i = threadIdx.x * PL + k;
            pref[i] = run;
            base[i] = v[k] && abl != 1 ? atomicAdd(&gcount[i], v[k]) : 0;
            run += v[k];
        }
        if (threadIdx.x == 31) total = run;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < PER; ++k)
        if (win[k] >= 0) {
            const int pos = pref[win[k]] + rk[k];
            sent[pos] = (b << 20) | (t0 + k * 512 + threadIdx.x);
            swin[pos] = (unsigned short)win[k];
        }
    __syncthreads();
    if (abl == 2) { if (threadIdx.x == 0 && sent[5] == 77) lists[0] = 1; return; }
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        const int w = swin[i];
        lists[(long)w * CAP + (base[w] + i - pref[w]) % CAP] = sent[i];
    }
}

void bucket_run(torch::Tensor pt, torch::Tensor seq, torch::Tensor gcount, torch::Tensor lists, int64_t ws, int64_t cap,
                int64_t ent, int64_t abl) {
    const int batch = seq.size(0), mp = pt.size(1);
    auto st = c10::cuda::getCurrentCUDAStream();
#define BK(E) bucket<E, 256><<<dim3((mp + E - 1) / E, batch), 512, 0, st>>>(pt.data_ptr<int>(), seq.data_ptr<int>(), \
        mp, (int)ws, (int)cap, gcount.data_ptr<int>(), lists.data_ptr<int>(), (int)abl)
    if (ent == 2048) BK(2048);
    else if (ent == 4096) BK(4096);
    else BK(8192);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// s114: atomic-free multi-split owner pass for physical-order decode full (gate of CLAUDE.md direction 3).
// p1: CTA = (slice s of ENT page-table positions, request b). Counting-sorts its entries by physical window
//     (phys / WS) in smem and writes them back IN PLACE of its own slice (u16 offsets within the window), plus its
//     exclusive per-window offsets off[slice][0..NW] (u16). No global atomics, all stores coalesced.
// mapb: consumer-side map build, CTA w = window w: for every slice reads its run for window w and sets
//     own[phys - w * WS] = b in smem (0xff = hole), then writes the map out (for checking) or a checksum.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

template <int ENT, int NW, int NT>
__global__ void __launch_bounds__(NT) p1(const int* __restrict__ pt, const int* __restrict__ seqs, int MP, int WS,
                                         int NSL, unsigned short* __restrict__ out, unsigned short* __restrict__ off) {
    __shared__ int hist[NW + 1];
    __shared__ unsigned short srt[ENT];
    const int s = blockIdx.x, b = blockIdx.y, t0 = s * ENT, slice = b * NSL + s;
    const int n = max(0, min(ENT, seqs[b] - t0));
    for (int i = threadIdx.x; i <= NW; i += NT) hist[i] = 0;
    __syncthreads();
    constexpr int PER = ENT / NT;
    int win[PER], rk[PER], loc[PER];
    const int* row = pt + (long)b * MP + t0;
#pragma unroll
    for (int k = 0; k < PER; ++k) {
        const int i = k * NT + threadIdx.x;
        win[k] = -1;
        if (i < n) {
            const int p = __ldg(row + i);
            win[k] = p / WS;
            loc[k] = p - win[k] * WS;
            rk[k] = atomicAdd(&hist[win[k]], 1);
        }
    }
    __syncthreads();
    // Exclusive scan of hist[0..NW) by warp 0 (NW / 32 bins per lane, NW padded to a multiple of 32 by the caller).
    if (threadIdx.x < 32) {
        constexpr int PL = (NW + 31) / 32;
        int v[PL], sum = 0;
#pragma unroll
        for (int k = 0; k < PL; ++k) {
            const int i = threadIdx.x * PL + k;
            v[k] = i < NW ? hist[i] : 0;
            sum += v[k];
        }
        int inc = sum;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffff, inc, o);
            if (threadIdx.x >= o) inc += y;
        }
        int run = inc - sum;
        __syncwarp();
#pragma unroll
        for (int k = 0; k < PL; ++k) {
            const int i = threadIdx.x * PL + k;
            if (i < NW) hist[i] = run;
            run += v[k];
        }
        if (threadIdx.x == 31) hist[NW] = run;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < PER; ++k)
        if (win[k] >= 0) srt[hist[win[k]] + rk[k]] = (unsigned short)loc[k];
    __syncthreads();
    // Coalesced stores: entries as u32 pairs, offsets row.
    unsigned* o32 = reinterpret_cast<unsigned*>(out + (long)slice * ENT);
    const unsigned* s32 = reinterpret_cast<const unsigned*>(srt);
    for (int i = threadIdx.x; i < (n + 1) / 2; i += NT) o32[i] = s32[i];
    for (int i = threadIdx.x; i <= NW; i += NT) off[(long)slice * (NW + 1) + i] = (unsigned short)hist[i];
}

template <int NW, int NT>
__global__ void __launch_bounds__(NT) mapb(const unsigned short* __restrict__ out, const unsigned short* __restrict__ off,
                                           int WS, int NSL, int nslices, int ENT, unsigned char* __restrict__ omap,
                                           int* __restrict__ sink, int write) {
    extern __shared__ unsigned char own[];
    const int w = blockIdx.x;
    for (int i = threadIdx.x; i < WS; i += NT) own[i] = 0xff;
    __syncthreads();
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    // Each warp takes slices wp, wp + NT/32, ...; lane 0..1 fetch the two offsets, then the run in 32-entry steps.
    for (int sl = wp; sl < nslices; sl += NT / 32) {
        const unsigned short* o = off + (long)sl * (NW + 1) + w;
        const int a = __ldg(o), e = __ldg(o + 1);
        const unsigned char bb = (unsigned char)(sl / NSL);
        const unsigned short* src = out + (long)sl * ENT;
        for (int i = a + lane; i < e; i += 32) own[__ldg(src + i)] = bb;
    }
    __syncthreads();
    if (write) {
        for (int i = threadIdx.x; i < WS; i += NT) omap[(long)w * WS + i] = own[i];
    } else {
        int z = 0;
        for (int i = threadIdx.x; i < WS; i += NT) z += own[i];
        if (z == 0x7fffffff) sink[0] = z;
    }
}

template <int NW>
__global__ void ptread(const int* __restrict__ pt, long n, int* sink) {
    int z = 0;
    for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n / 4; i += (long)gridDim.x * blockDim.x) {
        const int4 v = __ldg(reinterpret_cast<const int4*>(pt) + i);
        z ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (z == 0x12345678) sink[0] = z;
}

void p1_run(torch::Tensor pt, torch::Tensor seq, torch::Tensor out, torch::Tensor off, int64_t ws, int64_t ent) {
    const int batch = seq.size(0), mp = pt.size(1);
    auto st = c10::cuda::getCurrentCUDAStream();
#define P1(E, T) { const int nsl = (mp + E - 1) / E; p1<E, 132, T><<<dim3(nsl, batch), T, 0, st>>>(pt.data_ptr<int>(), \
        seq.data_ptr<int>(), mp, (int)ws, nsl, (unsigned short*)out.data_ptr(), (unsigned short*)off.data_ptr()); }
    if (ent == 2048) P1(2048, 512)
    else if (ent == 4096) P1(4096, 512)
    else if (ent == 8192) P1(8192, 1024)
    else P1(16384, 1024)
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void mapb_run(torch::Tensor out, torch::Tensor off, torch::Tensor omap, torch::Tensor sink, int64_t ws, int64_t nsl,
              int64_t nslices, int64_t ent, int64_t write, int64_t nt) {
    auto st = c10::cuda::getCurrentCUDAStream();
    if (nt == 512) {
        cudaFuncSetAttribute(mapb<132, 512>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)ws);
        mapb<132, 512><<<132, 512, ws, st>>>((unsigned short*)out.data_ptr(), (unsigned short*)off.data_ptr(), (int)ws,
            (int)nsl, (int)nslices, (int)ent, omap.data_ptr<unsigned char>(), sink.data_ptr<int>(), (int)write);
    } else {
        cudaFuncSetAttribute(mapb<132, 1024>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)ws);
        mapb<132, 1024><<<132, 1024, ws, st>>>((unsigned short*)out.data_ptr(), (unsigned short*)off.data_ptr(), (int)ws,
            (int)nsl, (int)nslices, (int)ent, omap.data_ptr<unsigned char>(), sink.data_ptr<int>(), (int)write);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ptread_run(torch::Tensor pt, torch::Tensor sink) {
    auto st = c10::cuda::getCurrentCUDAStream();
    ptread<132><<<132 * 4, 512, 0, st>>>(pt.data_ptr<int>(), pt.numel(), sink.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// v2: 4 entries per int4 load, NT threads x Q int4 per thread (ENT = 4 * Q * NT).
template <int Q, int NW, int NT>
__global__ void __launch_bounds__(NT) p1v(const int* __restrict__ pt, const int* __restrict__ seqs, int MP, int WS,
                                          int NSL, unsigned short* __restrict__ out, unsigned short* __restrict__ off) {
    constexpr int ENT = 4 * Q * NT;
    __shared__ int hist[NW + 1];
    __shared__ __align__(16) unsigned short srt[ENT];
    const int s = blockIdx.x, b = blockIdx.y, t0 = s * ENT, slice = b * NSL + s;
    const int n = max(0, min(ENT, seqs[b] - t0));
    for (int i = threadIdx.x; i <= NW; i += NT) hist[i] = 0;
    const int* row = pt + (long)b * MP + t0;
    int4 v[Q];
#pragma unroll
    for (int k = 0; k < Q; ++k) {
        const int i = (k * NT + threadIdx.x) * 4;
        // MP need not be a multiple of 4: scalar loads at the row's tail.
        if (i + 3 < n && ((MP & 3) == 0)) v[k] = __ldg(reinterpret_cast<const int4*>(row + i));
        else {
            v[k].x = i < n ? __ldg(row + i) : -1; v[k].y = i + 1 < n ? __ldg(row + i + 1) : -1;
            v[k].z = i + 2 < n ? __ldg(row + i + 2) : -1; v[k].w = i + 3 < n ? __ldg(row + i + 3) : -1;
        }
    }
    __syncthreads();
    int win[4 * Q], rk[4 * Q];
#pragma unroll
    for (int k = 0; k < Q; ++k) {
        const int p[4] = {v[k].x, v[k].y, v[k].z, v[k].w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            win[4 * k + j] = p[j] >= 0 ? p[j] / WS : -1;
            if (p[j] >= 0) rk[4 * k + j] = atomicAdd(&hist[win[4 * k + j]], 1);
        }
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        constexpr int PL = (NW + 31) / 32;
        int c[PL], sum = 0;
#pragma unroll
        for (int k = 0; k < PL; ++k) {
            const int i = threadIdx.x * PL + k;
            c[k] = i < NW ? hist[i] : 0;
            sum += c[k];
        }
        int inc = sum;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffff, inc, o);
            if (threadIdx.x >= o) inc += y;
        }
        int run = inc - sum;
        __syncwarp();
#pragma unroll
        for (int k = 0; k < PL; ++k) {
            const int i = threadIdx.x * PL + k;
            if (i < NW) hist[i] = run;
            run += c[k];
        }
        if (threadIdx.x == 31) hist[NW] = run;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < Q; ++k) {
        const int p[4] = {v[k].x, v[k].y, v[k].z, v[k].w};
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (p[j] >= 0) srt[hist[win[4 * k + j]] + rk[4 * k + j]] = (unsigned short)(p[j] - win[4 * k + j] * WS);
    }
    __syncthreads();
    uint4* o16 = reinterpret_cast<uint4*>(out + (long)slice * ENT);
    const uint4* s16 = reinterpret_cast<const uint4*>(srt);
    for (int i = threadIdx.x; i < (n + 7) / 8; i += NT) o16[i] = s16[i];
    for (int i = threadIdx.x; i <= NW; i += NT) off[(long)slice * (NW + 1) + i] = (unsigned short)hist[i];
}

// v2 map build: offsets of all slices into smem first (one parallel round), then each warp walks its slices with
// R entries per lane loaded before any store (independent loads), a tail loop for longer runs.
template <int NW, int NT, int R>
__global__ void __launch_bounds__(NT) mapbv(const unsigned short* __restrict__ out, const unsigned short* __restrict__ off,
                                            int WS, int NSL, int nslices, int ENT, unsigned char* __restrict__ omap,
                                            int* __restrict__ sink, int write) {
    extern __shared__ __align__(16) unsigned char sm[];
    int* sa = reinterpret_cast<int*>(sm);              // nslices (start, end) pairs packed
    unsigned char* own = sm + 4 * nslices;
    const int w = blockIdx.x;
    for (int i = threadIdx.x; i < nslices; i += NT) {
        const unsigned short* o = off + (long)i * (NW + 1) + w;
        sa[i] = (int)__ldg(o) | ((int)__ldg(o + 1) << 16);
    }
    for (int i = threadIdx.x; i < WS; i += NT) own[i] = 0xff;
    __syncthreads();
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
    constexpr int NWP = NT / 32;
    constexpr int U = 4;  // slices per batch
    for (int sl0 = wp * U; sl0 < nslices; sl0 += NWP * U) {
        unsigned short e[U][R];
        int st[U], en[U];
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const int sl = sl0 + u;
            const int pr = sl < nslices ? sa[sl] : 0;
            st[u] = pr & 0xffff; en[u] = pr >> 16;
            const unsigned short* src = out + (long)sl * ENT;
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int i = st[u] + r * 32 + lane;
                e[u][r] = i < en[u] ? __ldg(src + i) : 0xffff;
            }
        }
#pragma unroll
        for (int u = 0; u < U; ++u) {
            const unsigned char bb = (unsigned char)((sl0 + u) / NSL);
#pragma unroll
            for (int r = 0; r < R; ++r)
                if (e[u][r] != 0xffff) own[e[u][r]] = bb;
            const unsigned short* src = out + (long)(sl0 + u) * ENT;
            for (int i = st[u] + R * 32 + lane; i < en[u]; i += 32) own[__ldg(src + i)] = bb;
        }
    }
    __syncthreads();
    if (write) {
        for (int i = threadIdx.x; i < WS; i += NT) omap[(long)w * WS + i] = own[i];
    } else {
        int z = 0;
        for (int i = threadIdx.x; i < WS; i += NT) z += own[i];
        if (z == 0x7fffffff) sink[0] = z;
    }
}

void p1v_run(torch::Tensor pt, torch::Tensor seq, torch::Tensor out, torch::Tensor off, int64_t ws, int64_t ent) {
    const int batch = seq.size(0), mp = pt.size(1);
    auto st = c10::cuda::getCurrentCUDAStream();
#define P1V(Q, T) { constexpr int E = 4 * Q * T; const int nsl = (mp + E - 1) / E; p1v<Q, 132, T><<<dim3(nsl, batch), T, 0, st>>>( \
        pt.data_ptr<int>(), seq.data_ptr<int>(), mp, (int)ws, nsl, (unsigned short*)out.data_ptr(), (unsigned short*)off.data_ptr()); }
    if (ent == 1024) P1V(1, 256)
    else if (ent == 2048) P1V(2, 256)
    else if (ent == 4096) P1V(4, 256)
    else P1V(4, 512)
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void mapbv_run(torch::Tensor out, torch::Tensor off, torch::Tensor omap, torch::Tensor sink, int64_t ws, int64_t nsl,
               int64_t nslices, int64_t ent, int64_t write, int64_t r) {
    auto st = c10::cuda::getCurrentCUDAStream();
    const int smem = (int)(ws + 4 * nslices + 16);
#define MB(R) { cudaFuncSetAttribute(mapbv<132, 1024, R>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
        mapbv<132, 1024, R><<<132, 1024, smem, st>>>((unsigned short*)out.data_ptr(), (unsigned short*)off.data_ptr(), (int)ws, \
            (int)nsl, (int)nslices, (int)ent, omap.data_ptr<unsigned char>(), sink.data_ptr<int>(), (int)write); }
    if (r == 1) MB(1) else if (r == 2) MB(2) else MB(4)
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// v3: lean per-entry path: coalesced scalar loads, w = umulhi(p, MG) >> SH (exact for p < 2^22, checked on host).
template <int PER, int NW, int NT>
__global__ void __launch_bounds__(NT) p1w(const int* __restrict__ pt, const int* __restrict__ seqs, int MP, int WS,
                                          unsigned MG, int SH, int NSL, unsigned short* __restrict__ out,
                                          unsigned short* __restrict__ off) {
    constexpr int ENT = PER * NT;
    __shared__ int hist[NW + 1];
    __shared__ __align__(16) unsigned short srt[ENT];
    const int s = blockIdx.x, b = blockIdx.y, t0 = s * ENT, slice = b * NSL + s;
    const int n = max(0, min(ENT, seqs[b] - t0));
    for (int i = threadIdx.x; i <= NW; i += NT) hist[i] = 0;
    const int* row = pt + (long)b * MP + t0 + threadIdx.x;
    int p[PER];
#pragma unroll
    for (int k = 0; k < PER; ++k) p[k] = k * NT + threadIdx.x < n ? __ldg(row + k * NT) : -1;
    __syncthreads();
    int code[PER];  // (window << 16 | rank), -1 = none
#pragma unroll
    for (int k = 0; k < PER; ++k) {
        if (p[k] >= 0) {
            const int w = __umulhi((unsigned)p[k], MG) >> SH;
            code[k] = (w << 16) | atomicAdd(&hist[w], 1);
            p[k] -= w * WS;
        } else code[k] = -1;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        constexpr int PL = (NW + 31) / 32;
        int c[PL], sum = 0;
#pragma unroll
        for (int k = 0; k < PL; ++k) {
            const int i = threadIdx.x * PL + k;
            c[k] = i < NW ? hist[i] : 0;
            sum += c[k];
        }
        int inc = sum;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffff, inc, o);
            if (threadIdx.x >= o) inc += y;
        }
        int run = inc - sum;
        __syncwarp();
#pragma unroll
        for (int k = 0; k < PL; ++k) {
            const int i = threadIdx.x * PL + k;
            if (i < NW) hist[i] = run;
            run += c[k];
        }
        if (threadIdx.x == 31) hist[NW] = run;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < PER; ++k)
        if (code[k] >= 0) srt[hist[code[k] >> 16] + (code[k] & 0xffff)] = (unsigned short)p[k];
    __syncthreads();
    uint4* o16 = reinterpret_cast<uint4*>(out + (long)slice * ENT);
    const uint4* s16 = reinterpret_cast<const uint4*>(srt);
    for (int i = threadIdx.x; i < (n + 7) / 8; i += NT) o16[i] = s16[i];
    for (int i = threadIdx.x; i <= NW; i += NT) off[(long)slice * (NW + 1) + i] = (unsigned short)hist[i];
}

// v3 map build: one LANE per slice (a warp covers 32 slices, loops to the longest run of the 32).
template <int NW, int NT>
__global__ void __launch_bounds__(NT) mapbl(const unsigned short* __restrict__ out, const unsigned short* __restrict__ off,
                                            int WS, int NSL, int nslices, int ENT, unsigned char* __restrict__ omap,
                                            int* __restrict__ sink, int write) {
    extern __shared__ __align__(16) unsigned char own[];
    const int w = blockIdx.x;
    for (int i = threadIdx.x; i < (WS + 3) / 4; i += NT) reinterpret_cast<unsigned*>(own)[i] = 0xffffffffu;
    __syncthreads();
    for (int sl = threadIdx.x; sl < nslices; sl += NT) {
        const unsigned short* o = off + (long)sl * (NW + 1) + w;
        const int a = __ldg(o), e = __ldg(o + 1);
        const unsigned char bb = (unsigned char)(sl / NSL);
        const unsigned short* src = out + (long)sl * ENT;
        // Aligned 16-B vectors (ENT is a multiple of 8, slices start 16-B aligned): one L1 tag per 8 entries.
        for (int i = a & ~7; i < e; i += 8) {
            const uint4 v = __ldg(reinterpret_cast<const uint4*>(src + i));
            const unsigned u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int k = i + j;
                if (k >= a && k < e) own[(u[j >> 1] >> (16 * (j & 1))) & 0xffff] = bb;
            }
        }
    }
    __syncthreads();
    if (write) {
        for (int i = threadIdx.x; i < WS; i += NT) omap[(long)w * WS + i] = own[i];
    } else {
        int z = 0;
        for (int i = threadIdx.x; i < WS; i += NT) z += own[i];
        if (z == 0x7fffffff) sink[0] = z;
    }
}

void p1w_run(torch::Tensor pt, torch::Tensor seq, torch::Tensor out, torch::Tensor off, int64_t ws, int64_t mg,
             int64_t sh, int64_t ent) {
    const int batch = seq.size(0), mp = pt.size(1);
    auto st = c10::cuda::getCurrentCUDAStream();
#define P1W(P, T) { constexpr int E = P * T; const int nsl = (mp + E - 1) / E; p1w<P, 132, T><<<dim3(nsl, batch), T, 0, st>>>( \
        pt.data_ptr<int>(), seq.data_ptr<int>(), mp, (int)ws, (unsigned)mg, (int)sh, nsl, (unsigned short*)out.data_ptr(), \
        (unsigned short*)off.data_ptr()); }
    if (ent == 1024) P1W(4, 256)
    else if (ent == 2048) P1W(8, 256)
    else if (ent == 4096) P1W(16, 256)
    else P1W(16, 512)
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void mapbl_run(torch::Tensor out, torch::Tensor off, torch::Tensor omap, torch::Tensor sink, int64_t ws, int64_t nsl,
               int64_t nslices, int64_t ent, int64_t write, int64_t nt) {
    auto st = c10::cuda::getCurrentCUDAStream();
#define MBL(T) { cudaFuncSetAttribute(mapbl<132, T>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)ws + 16); \
        mapbl<132, T><<<132, T, ws + 16, st>>>((unsigned short*)out.data_ptr(), (unsigned short*)off.data_ptr(), (int)ws, \
            (int)nsl, (int)nslices, (int)ent, omap.data_ptr<unsigned char>(), sink.data_ptr<int>(), (int)write); }
    if (nt == 256) MBL(256) else if (nt == 512) MBL(512) else MBL(1024)
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

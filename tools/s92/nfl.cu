// s92: batch-1 decode FULL attention as a native kernel with a pipelined per-warp loop (csl's mma.sync E4M3 tile).
// Grid: NCTA CTAs x NW warps; warp w handles KV head w & 1 and pair p = w >> 1 of its CTA's key chunks (32 keys each,
// interleaved across pairs, so both heads' halves of a 128-B row are read on the same SM). Each warp keeps DEP chunks of
// K/V in flight in registers (page-table entries 2 DEP chunks ahead, one coalesced 32-entry load per chunk, spread by
// shuffles), online softmax in FP32, P in E4M3 with FA3's max offset 8. The NW/2 warps of each head merge in shared
// memory; the CTA writes one FP32 partial per (head, query row) = slot blockIdx.x of _combine's layout.
// Fixed shapes: HQ 16, HKV 2, D 64, batch 1, L 1, page 1.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

#ifndef NFL_NW
#define NFL_NW 4
#endif
#ifndef NFL_F16
#define NFL_F16 1
#endif
#ifndef NFL_CPA
#define NFL_CPA 1
#endif
#ifndef NFL_DEP
#define NFL_DEP 3
#endif
namespace nfl {
constexpr int HQ = 16, HKV = 2, D = 64, QKV_W = (2 * HQ + 2 * HKV) * D;

__device__ __forceinline__ unsigned pk8(float hi, float lo) {
    unsigned short r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
    return r;
}
__device__ __forceinline__ void mma8(float* d, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                     unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// E4M3 pair -> FP16 pair (exact): low byte -> low half.
__device__ __forceinline__ unsigned h2(unsigned x) {
    unsigned r;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"((unsigned short)x));
    return r;
}
__device__ __forceinline__ void mma16(float* d, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                      unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float bf(unsigned short x) { return __uint_as_float((unsigned)x << 16); }

template <int NW, int DEP>
__global__ void __launch_bounds__(NW * 32) decode(const unsigned short* __restrict__ qkv, unsigned char* kc,
                                                  unsigned char* vc, const int* __restrict__ pt,
                                                  const int* __restrict__ seqs, const long* __restrict__ loc,
                                                  const float* __restrict__ ks, const float* __restrict__ vs,
                                                  float* __restrict__ pacc, float* __restrict__ pml, int MP,
                                                  float qk_scale) {
    constexpr int NP = NW / 2;
    __shared__ __align__(16) float lpart[NW][8][D];
    __shared__ __align__(16) float lml[NW][8][2];
    __shared__ __align__(16) unsigned char newkv[2][2][D];
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
    const int h = warp & 1, p = warp >> 1;
    // Keys [0, MP): MP = the table width >= seq + 1 (SGLang slices it to the longest length), so the chunks are known
    // without waiting for seq_lens; keys past seq are masked.
    const int nct = (MP + 31) >> 5;
    const int per = (nct + gridDim.x - 1) / gridDim.x;
    const int c0 = blockIdx.x * per, c1 = min(c0 + per, nct);
    const int nch = c1 > c0 + p ? (c1 - c0 - p + NP - 1) / NP : 0;
    // Padded to a multiple of DEP so the unrolled loop has no early exit (shuffles stay on the converged path);
    // the padding chunks read row 0 and are fully masked.
    const int ncp = (nch + DEP - 1) / DEP * DEP;

    int ptv[DEP];
    uint4 kr[DEP][4];
    uint2 vr[DEP][8];
#if NFL_CPA
    // cp.async ring per warp (register-pipelined loads share the warp's few scoreboards, so each chunk waited a
    // whole memory trip): slot = K 4 x 32 x 16 B + V 8 x 32 x 8 B, each lane's own bytes, plus 32 page entries.
    // Every location is written and read by the same lane: no warp barriers.
    extern __shared__ __align__(16) unsigned char dsm[];
    constexpr int SLOT = 4096;
    const unsigned sb = (unsigned)__cvta_generic_to_shared(dsm) + warp * DEP * (SLOT + 128);
#define NFL_PT(s, k)                                                                                           \
    {                                                                                                          \
        const int key = 32 * (c0 + p + (k) * NP) + lane;                                                       \
        const int* src = pt + ((k) < nch && key < MP ? key : 0);                                               \
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(sb + DEP * SLOT + (s) * 128 + 4 * lane),  \
                     "l"(src) : "memory");                                                                     \
    }
#define NFL_LDE(s, e)                                                                                          \
    {                                                                                                          \
        _Pragma("unroll") for (int j = 0; j < 4; ++j) {                                                        \
            const unsigned sl = max(__shfl_sync(0xffffffff, e, 8 * j + g), 0);                                \
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(sb + (s) * SLOT + (32 * j + lane) * 16), \
                         "l"(kc + (sl * (HKV * D) + h * D + 16 * c)) : "memory");                              \
        }                                                                                                      \
        _Pragma("unroll") for (int i = 0; i < 8; ++i) {                                                        \
            const unsigned sl = max(__shfl_sync(0xffffffff, e, 8 * (i >> 1) + 2 * c + (i & 1)), 0);           \
            asm volatile("cp.async.ca.shared.global [%0], [%1], 8;"                                           \
                         ::"r"(sb + (s) * SLOT + 2048 + (32 * i + lane) * 8),                                  \
                         "l"(vc + (sl * (HKV * D) + h * D + 8 * g)) : "memory");                               \
        }                                                                                                      \
    }
#pragma unroll
    for (int s = 0; s < DEP; ++s) {
        const int key = 32 * (c0 + p + s * NP) + lane;
        ptv[s] = s < nch && key < MP ? pt[key] : 0;
    }
#pragma unroll
    for (int s = 0; s < DEP; ++s) {
        NFL_LDE(s, ptv[s])
        NFL_PT(s, s + DEP)
        asm volatile("cp.async.commit_group;" ::: "memory");
    }
#else
#define NFL_PT(s, k)                                                                                           \
    {                                                                                                          \
        const int key = 32 * (c0 + p + (k) * NP) + lane;                                                       \
        ptv[s] = (k) < nch && key < MP ? pt[key] : 0;                                                          \
    }
#define NFL_LD(s)                                                                                              \
    {                                                                                                          \
        _Pragma("unroll") for (int j = 0; j < 4; ++j) {                                                        \
            const unsigned sl = max(__shfl_sync(0xffffffff, ptv[s], 8 * j + g), 0);                           \
            kr[s][j] = *(const uint4*)(kc + (sl * (HKV * D) + h * D + 16 * c));                                \
        }                                                                                                      \
        _Pragma("unroll") for (int i = 0; i < 8; ++i) {                                                        \
            const unsigned sl = max(__shfl_sync(0xffffffff, ptv[s], 8 * (i >> 1) + 2 * c + (i & 1)), 0);      \
            vr[s][i] = *(const uint2*)(vc + (sl * (HKV * D) + h * D + 8 * g));                                 \
        }                                                                                                      \
    }
#pragma unroll
    for (int s = 0; s < DEP; ++s) NFL_PT(s, s)
#pragma unroll
    for (int s = 0; s < DEP; ++s) {
        NFL_LD(s)
        NFL_PT(s, s + DEP)
    }
#endif

    // Queries (row g = query head 8 h + g, d 16 c .. 16 c + 15): RMSNorm, E4M3.
    const unsigned short* qp = qkv + (8 * h + g) * D + 16 * c;
    const uint4 qa = *(const uint4*)qp, qb = *(const uint4*)(qp + 8);
    const unsigned short* kp = qkv + (HQ + h) * D + 2 * lane;
    const unsigned kw = *(const unsigned*)kp, vw = *(const unsigned*)(kp + HKV * D);
    const int seq = seqs[0];
    const float kscale = *ks, vscale = *vs;
    unsigned q8[4];
    {
        float x[16];
        unsigned u[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) { x[2 * i] = bf(u[i] & 0xffff); x[2 * i + 1] = bf(u[i] >> 16); }
        float ss = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) ss += x[i] * x[i];
        ss += __shfl_xor_sync(0xffffffff, ss, 1);
        ss += __shfl_xor_sync(0xffffffff, ss, 2);
        const float inv = rsqrtf(ss / D + 1e-5f);
#pragma unroll
        for (int w = 0; w < 4; ++w)
            q8[w] = pk8(x[4 * w + 1] * inv, x[4 * w] * inv) | (pk8(x[4 * w + 3] * inv, x[4 * w + 2] * inv) << 16);
    }
#if NFL_F16
    // FP16 MMAs on E4M3 values (exact), since sm_90 emulates the E4M3 mma.sync with extra conversions. The d order
    // inside each 16-byte chunk is permuted the same way for Q and K: step t takes bytes 4t .. 4t + 3.
    unsigned qh[8];
#pragma unroll
    for (int w = 0; w < 4; ++w) { qh[2 * w] = h2(q8[w] & 0xffff); qh[2 * w + 1] = h2(q8[w] >> 16); }
#endif
    // The new token (key seq): the warp whose chunk holds it normalizes it, writes the cache slot and patches it in.
    const int own = seq >> 5;  // chunk index of key seq
    const bool newer = own >= c0 && own < c1 && (own - c0) % NP == p;
    if (newer) {
        float k0 = bf(kw & 0xffff), k1 = bf(kw >> 16);
        float s = k0 * k0 + k1 * k1;
#pragma unroll
        for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
        const float ki = rsqrtf(s / D + 1e-5f) / kscale;
        const unsigned short k8 = pk8(k1 * ki, k0 * ki);
        const unsigned short v8 = pk8(bf(vw >> 16) / vscale, bf(vw & 0xffff) / vscale);
        *(unsigned short*)&newkv[h][0][2 * lane] = k8;
        *(unsigned short*)&newkv[h][1][2 * lane] = v8;
        const long slot = loc[0] * (HKV * D) + h * D + 2 * lane;
        *(unsigned short*)(kc + slot) = k8;
        *(unsigned short*)(vc + slot) = v8;
        __syncwarp();
    }
    const float scale = qk_scale * kscale;
    float m_run = -1e30f, l_run = 0.f;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;

    for (int k0 = 0; k0 < ncp; k0 += DEP) {
#pragma unroll
        for (int s = 0; s < DEP; ++s) {
            const int k = k0 + s;
            const int kb = k < nch ? 32 * (c0 + p + k * NP) : (1 << 30);
#if NFL_CPA
            asm volatile("cp.async.wait_group %0;" ::"n"(DEP - 1) : "memory");
#pragma unroll
            for (int j = 0; j < 4; ++j)
                asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                             : "=r"(kr[s][j].x), "=r"(kr[s][j].y), "=r"(kr[s][j].z), "=r"(kr[s][j].w)
                             : "r"(sb + s * SLOT + (32 * j + lane) * 16) : "memory");
#pragma unroll
            for (int i = 0; i < 8; ++i)
                asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(vr[s][i].x), "=r"(vr[s][i].y)
                             : "r"(sb + s * SLOT + 2048 + (32 * i + lane) * 8) : "memory");
            int e;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(e) : "r"(sb + DEP * SLOT + s * 128 + 4 * lane) : "memory");
#endif
            if (newer && (seq >> 5) == (kb >> 5)) {
                const int kk = seq - kb;
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    if (8 * j + g == kk) kr[s][j] = *(const uint4*)&newkv[h][0][16 * c];
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    if (8 * (i >> 1) + 2 * c + (i & 1) == kk) vr[s][i] = *(const uint2*)&newkv[h][1][8 * g];
            }
            // S = Q K^T for row g: tile j holds keys 8j + 2c + {0, 1}.
            float sc[4][4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.f;
#if NFL_F16
                const unsigned kw4[4] = {kr[s][j].x, kr[s][j].y, kr[s][j].z, kr[s][j].w};
#pragma unroll
                for (int t = 0; t < 4; ++t)
                    mma16(sc[j], qh[2 * t], 0u, qh[2 * t + 1], 0u, h2(kw4[t] & 0xffff), h2(kw4[t] >> 16));
#else
                mma8(sc[j], q8[0], 0u, q8[1], 0u, kr[s][j].x, kr[s][j].y);
                mma8(sc[j], q8[2], 0u, q8[3], 0u, kr[s][j].z, kr[s][j].w);
#endif
            }
            float mc = -1e30f;
            if (kb + 32 <= seq + 1) {
#pragma unroll
                for (int j = 0; j < 4; ++j)
#pragma unroll
                    for (int e = 0; e < 2; ++e) { sc[j][e] *= scale; mc = fmaxf(mc, sc[j][e]); }
            } else {
#pragma unroll
                for (int j = 0; j < 4; ++j)
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                        sc[j][e] = kb + 8 * j + 2 * c + e <= seq ? sc[j][e] * scale : -INFINITY;
                        mc = fmaxf(mc, sc[j][e]);
                    }
            }
            mc = fmaxf(mc, __shfl_xor_sync(0xffffffff, mc, 1));
            mc = fmaxf(mc, __shfl_xor_sync(0xffffffff, mc, 2));
            const float mn = fmaxf(m_run, mc);
            const float alpha = exp2f(m_run - mn);
            m_run = mn;
            const float mo = mn - 8.f;  // FA3's FP8 max offset: P <= 2^8
            float ls = 0.f;
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int e = 0; e < 2; ++e) { sc[j][e] = exp2f(sc[j][e] - mo); ls += sc[j][e]; }
            l_run = l_run * alpha + ls;  // per-thread share of row g's sum (reduced over c at the end)
#if NFL_F16
            unsigned ph[4];  // P rounded to E4M3 (FA3's P@V operand), held as FP16
#pragma unroll
            for (int j = 0; j < 4; ++j) ph[j] = h2(pk8(sc[j][1], sc[j][0]));
#else
            const unsigned pb0 = pk8(sc[0][1], sc[0][0]) | (pk8(sc[1][1], sc[1][0]) << 16);
            const unsigned pb1 = pk8(sc[2][1], sc[2][0]) | (pk8(sc[3][1], sc[3][0]) << 16);
#endif
            // acc columns are query rows 2c, 2c + 1: their alphas live on lanes 8c and 8c + 4.
            const float a0 = __shfl_sync(0xffffffff, alpha, 8 * c), a1 = __shfl_sync(0xffffffff, alpha, 8 * c + 4);
            // O^T += V^T P^T: m-tile i holds d 8g + 2i (rows m = g) and 8g + 2i + 1 (m = g + 8).
#if NFL_F16
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc[i][0] *= a0; acc[i][2] *= a0; acc[i][1] *= a1; acc[i][3] *= a1;
                const int bsel = (i & 1) ? 0x7362 : 0x5140;
#pragma unroll
                for (int u = 0; u < 2; ++u) {
                    const unsigned w0 = i < 2 ? vr[s][4 * u].x : vr[s][4 * u].y;
                    const unsigned w1 = i < 2 ? vr[s][4 * u + 1].x : vr[s][4 * u + 1].y;
                    const unsigned w2 = i < 2 ? vr[s][4 * u + 2].x : vr[s][4 * u + 2].y;
                    const unsigned w3 = i < 2 ? vr[s][4 * u + 3].x : vr[s][4 * u + 3].y;
                    const unsigned t01 = __byte_perm(w0, w1, bsel), t23 = __byte_perm(w2, w3, bsel);
                    mma16(acc[i], h2(t01 & 0xffff), h2(t01 >> 16), h2(t23 & 0xffff), h2(t23 >> 16), ph[2 * u],
                          ph[2 * u + 1]);
                }
            }
#else
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int bsel = (i & 1) ? 0x7362 : 0x5140;
                unsigned w[8];
#pragma unroll
                for (int q = 0; q < 8; ++q) w[q] = (i < 2) ? vr[s][q].x : vr[s][q].y;
                unsigned t01 = __byte_perm(w[0], w[1], bsel), t23 = __byte_perm(w[2], w[3], bsel);
                unsigned t45 = __byte_perm(w[4], w[5], bsel), t67 = __byte_perm(w[6], w[7], bsel);
                acc[i][0] *= a0; acc[i][2] *= a0; acc[i][1] *= a1; acc[i][3] *= a1;
                mma8(acc[i], __byte_perm(t01, t23, 0x5410), __byte_perm(t01, t23, 0x7632),
                     __byte_perm(t45, t67, 0x5410), __byte_perm(t45, t67, 0x7632), pb0, pb1);
            }
#endif
#if NFL_CPA
            NFL_LDE(s, e)
            NFL_PT(s, k + 2 * DEP)
            asm volatile("cp.async.commit_group;" ::: "memory");
#else
            NFL_LD(s)
            NFL_PT(s, k + 2 * DEP)
#endif
        }
    }
#undef NFL_LD
#undef NFL_LDE
#undef NFL_PT
    l_run += __shfl_xor_sync(0xffffffff, l_run, 1);
    l_run += __shfl_xor_sync(0xffffffff, l_run, 2);
    {
        float* lp = &lpart[warp][0][0];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            float* r = lp + (2 * c + e) * D + 8 * g;
            *(float4*)r = make_float4(acc[0][e], acc[0][e + 2], acc[1][e], acc[1][e + 2]);
            *(float4*)(r + 4) = make_float4(acc[2][e], acc[2][e + 2], acc[3][e], acc[3][e + 2]);
        }
        if (c == 0) *(float2*)&lml[warp][g][0] = make_float2(m_run - 8.f, l_run);
    }
    __syncthreads();
    // Merge the NP warps of each head: work item = (head, row, 4 d); write slot blockIdx.x.
    const int S = gridDim.x;
    for (int t = tid; t < 2 * 8 * 16; t += NW * 32) {
        const int h2 = t >> 7, row = (t >> 4) & 7, dc = t & 15;
        float mk[NP], M = -1e30f;
#pragma unroll
        for (int k = 0; k < NP; ++k) { mk[k] = lml[2 * k + h2][row][0]; M = fmaxf(M, mk[k]); }
        float L = 0.f, o[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int k = 0; k < NP; ++k) {
            const float w = exp2f(mk[k] - M);
            L += w * lml[2 * k + h2][row][1];
            const float4 a = *(const float4*)&lpart[2 * k + h2][row][4 * dc];
            o[0] += w * a.x; o[1] += w * a.y; o[2] += w * a.z; o[3] += w * a.w;
        }
        const long r = ((long)h2 * S + blockIdx.x) * 8 + row;
        *(float4*)(pacc + r * D + 4 * dc) = make_float4(o[0], o[1], o[2], o[3]);
        if (dc == 0) *(float2*)(pml + r * 2) = make_float2(M, L);
    }
}
}  // namespace nfl

void nfl_decode(torch::Tensor qkv, torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seq,
                torch::Tensor loc, torch::Tensor ks, torch::Tensor vs, torch::Tensor pacc, torch::Tensor pml,
                int64_t ncta, double qk_scale) {
    TORCH_CHECK(kc.numel() < (1LL << 32) && vc.numel() < (1LL << 32), "nfl: 32-bit cache offsets");
    const int smem = NFL_CPA ? NFL_NW * NFL_DEP * (4096 + 128) : 0;
    static bool set = false;
    if (!set) {
        cudaFuncSetAttribute(nfl::decode<NFL_NW, NFL_DEP>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        set = true;
    }
    nfl::decode<NFL_NW, NFL_DEP><<<(int)ncta, NFL_NW * 32, smem, c10::cuda::getCurrentCUDAStream()>>>(
        (const unsigned short*)qkv.data_ptr(), (unsigned char*)kc.data_ptr(), (unsigned char*)vc.data_ptr(),
        pt.data_ptr<int>(), seq.data_ptr<int>(), loc.data_ptr<long>(), ks.data_ptr<float>(), vs.data_ptr<float>(),
        pacc.data_ptr<float>(), pml.data_ptr<float>(), (int)pt.size(1), (float)qk_scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

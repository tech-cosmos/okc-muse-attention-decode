#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
// Consumer-only microbenchmark for the verify-full attention consumer (one warpgroup per KV head).
// K/V tiles are preloaded into shared memory once; each CTA then runs NT tiles over the NS slots.
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

constexpr int BN = 64, HD = 64, MR = 64;

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint64_t mdesc(uint32_t addr, uint32_t sbo, uint32_t mode) {
    return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)(sbo >> 4) << 32) | ((uint64_t)mode << 62);
}
__device__ __forceinline__ uint32_t swz128(uint32_t o) { return o ^ (((o >> 7) & 7) << 4); }
__device__ __forceinline__ uint32_t swz64(uint32_t o) { return o ^ (((o >> 7) & 3) << 4); }

__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N> __device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"); }
template <int N> __device__ __forceinline__ void fence_regs(float (&r)[N]) {
#pragma unroll
    for (int i = 0; i < N; ++i) asm volatile("" : "+f"(r[i])::"memory");
}
template <int N> __device__ __forceinline__ void fence_regs(uint32_t (&r)[N]) {
#pragma unroll
    for (int i = 0; i < N; ++i) asm volatile("" : "+r"(r[i])::"memory");
}

#define D32(a) "+f"(a[0]), "+f"(a[1]), "+f"(a[2]), "+f"(a[3]), "+f"(a[4]), "+f"(a[5]), "+f"(a[6]), "+f"(a[7]), \
    "+f"(a[8]), "+f"(a[9]), "+f"(a[10]), "+f"(a[11]), "+f"(a[12]), "+f"(a[13]), "+f"(a[14]), "+f"(a[15]), \
    "+f"(a[16]), "+f"(a[17]), "+f"(a[18]), "+f"(a[19]), "+f"(a[20]), "+f"(a[21]), "+f"(a[22]), "+f"(a[23]), \
    "+f"(a[24]), "+f"(a[25]), "+f"(a[26]), "+f"(a[27]), "+f"(a[28]), "+f"(a[29]), "+f"(a[30]), "+f"(a[31])
#define R32 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}"

// S[64 x 64] (+)= Q[64 x 32 fp8] K^T, one k32 step.
__device__ __forceinline__ void qk_k32(float (&s)[32], uint64_t da, uint64_t db, int acc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 " R32 ", %32, %33, p, 1, 1;\n}\n"
                 : D32(s) : "l"(da), "l"(db), "r"(acc));
}
// O[64 x 64] += P[64 x 16 fp16, registers] V[16 x 64 fp16, MN-major smem], one k16 step.
__device__ __forceinline__ void pv_k16(float (&o)[32], const uint32_t (&a)[4], uint64_t db) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 " R32 ", {%32,%33,%34,%35}, %36, 1, 1, 1, 1;\n"
                 : D32(o) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db));
}
__device__ __forceinline__ void pv1_k16(float (&o)[4], const uint32_t (&a)[4], uint64_t db) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0,%1,%2,%3}, {%4,%5,%6,%7}, %8, 1, 1, 1, 1;\n"
                 : "+f"(o[0]), "+f"(o[1]), "+f"(o[2]), "+f"(o[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db));
}
// O[64 x 64] += P[64 x 32 e4m3, registers] V^T[64 x 32 e4m3, K-major smem], one k32 step.
__device__ __forceinline__ void pv8_k32(float (&o)[32], const uint32_t (&a)[4], uint64_t db) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 " R32 ", {%32,%33,%34,%35}, %36, 1, 1, 1;\n"
                 : D32(o) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db));
}
__device__ __forceinline__ uint32_t pack_e4(float a, float b, float c, float d) {
    uint16_t lo, hi;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(lo) : "f"(a), "f"(b));
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(hi) : "f"(c), "f"(d));
    return (uint32_t)lo | ((uint32_t)hi << 16);
}
#ifndef PV8
#define PV8 0
#endif
__device__ __forceinline__ uint32_t pack_h2(float a, float b) {
    __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ float ex2p(float x) {
    x = fmaxf(x, -126.f);
    float t = x + 12582912.f, f = x - (t - 12582912.f);
    float p = fmaf(fmaf(fmaf(0.0555041086648216f, f, 0.2402264923172484f), f, 0.6931471805599453f), f, 1.0f);
    return __int_as_float(__float_as_int(p) + (__float_as_int(t) << 23));
}
#ifndef ONES
#define ONES 0
#endif
#ifndef SPEC
#define SPEC 0
#endif
#ifndef LAZY
#define LAZY 0
#endif
#ifndef NPOLY
#define NPOLY 0
#endif
__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }

template <int NS> struct Smem {
    alignas(1024) uint8_t k[NS][BN * 128];      // NS slots, rows of both heads' 64 FP8 bytes, SW128
    alignas(1024) __half v[NS][2][BN * HD];       // per slot per head FP16 [key][d], SW128 (MN-major)
    alignas(1024) uint8_t q[2][MR * HD];          // per head FP8 [row][d], SW64
    alignas(1024) __half ones[2048];
    alignas(1024) uint8_t vt[NS][2][HD * BN];     // per slot per head FP8 V^T [d][key (permuted)], SW64
};

// Issue QK^T for one tile (2 k32 steps).
__device__ __forceinline__ void issue_qk(float (&s)[32], uint32_t qa, uint32_t ka) {
    wg_fence();
    qk_k32(s, mdesc(qa, 512, 2), mdesc(ka, 1024, 1), 0);
    qk_k32(s, mdesc(qa + 32, 512, 2), mdesc(ka + 32, 1024, 1), 1);
    wg_commit();
}
__device__ float g_lsum[4];
__device__ __forceinline__ void issue_pv(float (&o)[32], const uint32_t (&p)[16], uint32_t va, float (&ls)[4], uint32_t onea) {
    wg_fence();
    if (PV8) {
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            uint32_t a[4] = {p[kk * 4 + 0], p[kk * 4 + 1], p[kk * 4 + 2], p[kk * 4 + 3]};
            pv8_k32(o, a, mdesc(va + kk * 32, 512, 2));
        }
        wg_commit();
        return;
    }
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
        uint32_t a[4] = {p[kk * 4 + 0], p[kk * 4 + 1], p[kk * 4 + 2], p[kk * 4 + 3]};
        pv_k16(o, a, mdesc(va + kk * 2048, 1024, 1));
        if (ONES) pv1_k16(ls, a, mdesc(onea + kk * 256, 128, 0));
    }
    wg_commit();
}
// Online softmax on S (rows r0 = regs 0,1 / r1 = regs 2,3 of each n8 block). Writes P (fp16 A fragments),
// returns alpha for both rows.
__device__ __forceinline__ void softmax(float (&s)[32], float (&m)[2], float (&l)[2], uint32_t (&p)[16],
                                        float (&alpha)[2], float scale) {
#if SPEC
    if (m[0] > -1e29f) {
        // Speculative: exps with the running maxima right away; the tile maxima and the vote run alongside, and only
        // a row growing by more than 8 (P would pass 2^8) redoes the tile with new maxima (rare after the first tiles).
        const float o0 = m[0], o1 = m[1];
        float e[32];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            e[j * 4 + 0] = ex2(fmaf(s[j * 4 + 0], scale, -o0)); e[j * 4 + 1] = ex2(fmaf(s[j * 4 + 1], scale, -o0));
            e[j * 4 + 2] = ex2(fmaf(s[j * 4 + 2], scale, -o1)); e[j * 4 + 3] = ex2(fmaf(s[j * 4 + 3], scale, -o1));
        }
#if SPEC == 2
        // No tile maxima at all: P >= 0, so every P <= 2^8 iff the thread's partial row sums are <= 2^8.
        float l0 = 0.f, l1 = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) { l0 += e[j * 4 + 0] + e[j * 4 + 1]; l1 += e[j * 4 + 2] + e[j * 4 + 3]; }
        bool grow = !(fmaxf(l0, l1) <= 256.f);
        if (!__any_sync(0xffffffff, grow)) {
            alpha[0] = 1.f; alpha[1] = 1.f;
#else
        float mx0 = s[0], mx1 = s[2];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            mx0 = fmaxf(mx0, fmaxf(s[j * 4 + 0], s[j * 4 + 1]));
            mx1 = fmaxf(mx1, fmaxf(s[j * 4 + 2], s[j * 4 + 3]));
        }
        bool grow = (mx0 * scale > o0 + 8.f) || (mx1 * scale > o1 + 8.f);
        if (!__any_sync(0xffffffff, grow)) {
            alpha[0] = 1.f; alpha[1] = 1.f;
            float l0 = 0.f, l1 = 0.f;
#pragma unroll
            for (int j = 0; j < 8; ++j) { l0 += e[j * 4 + 0] + e[j * 4 + 1]; l1 += e[j * 4 + 2] + e[j * 4 + 3]; }
#endif
#pragma unroll
            if (PV8) {
                for (int kk = 0; kk < 2; ++kk) {
                    const int b0 = 16 * kk;
                    p[kk * 4 + 0] = pack_e4(e[b0 + 0], e[b0 + 1], e[b0 + 4], e[b0 + 5]);
                    p[kk * 4 + 1] = pack_e4(e[b0 + 2], e[b0 + 3], e[b0 + 6], e[b0 + 7]);
                    p[kk * 4 + 2] = pack_e4(e[b0 + 8], e[b0 + 9], e[b0 + 12], e[b0 + 13]);
                    p[kk * 4 + 3] = pack_e4(e[b0 + 10], e[b0 + 11], e[b0 + 14], e[b0 + 15]);
                }
            } else {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    p[(j >> 1) * 4 + (j & 1) * 2 + 0] = pack_h2(e[j * 4 + 0], e[j * 4 + 1]);
                    p[(j >> 1) * 4 + (j & 1) * 2 + 1] = pack_h2(e[j * 4 + 2], e[j * 4 + 3]);
                }
            }
            l[0] += l0; l[1] += l1;
            return;
        }
    }
#endif
    float mx0 = s[0], mx1 = s[2];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        mx0 = fmaxf(mx0, fmaxf(s[j * 4 + 0], s[j * 4 + 1]));
        mx1 = fmaxf(mx1, fmaxf(s[j * 4 + 2], s[j * 4 + 3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, 2));
    float mn0 = fmaxf(m[0], mx0 * scale), mn1 = fmaxf(m[1], mx1 * scale);
    if (LAZY) {
        // FA4: keep the old maxima unless one of the warp's rows grew by more than 8 (P <= 2^8).
        bool grow = (mn0 > m[0] + 8.f) || (mn1 > m[1] + 8.f);
        if (!__any_sync(0xffffffff, grow)) { mn0 = m[0]; mn1 = m[1]; }
    }
    alpha[0] = ex2(m[0] - mn0); alpha[1] = ex2(m[1] - mn1);
    m[0] = mn0; m[1] = mn1;
    float l0 = 0.f, l1 = 0.f;
    if (PV8) {
        // P * 2^8 (FA3's FP8 max offset) as E4M3; k32 step kk takes n8 blocks 4kk..4kk+3: regs (r0: blocks 4kk,
        // 4kk+1), (r1: same), (r0: blocks 4kk+2, 4kk+3), (r1: same). V^T's keys are permuted to match.
        const float off = LAZY ? 0.f : 8.f;  // lazy maxima already allow P up to 2^8
        const float o0 = mn0 - off, o1 = mn1 - off;
        float e[32];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            e[j * 4 + 0] = ex2(fmaf(s[j * 4 + 0], scale, -o0)); e[j * 4 + 1] = ex2(fmaf(s[j * 4 + 1], scale, -o0));
            e[j * 4 + 2] = ex2(fmaf(s[j * 4 + 2], scale, -o1)); e[j * 4 + 3] = ex2(fmaf(s[j * 4 + 3], scale, -o1));
            l0 += e[j * 4 + 0] + e[j * 4 + 1]; l1 += e[j * 4 + 2] + e[j * 4 + 3];
        }
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            const int b0 = 16 * kk;
            p[kk * 4 + 0] = pack_e4(e[b0 + 0], e[b0 + 1], e[b0 + 4], e[b0 + 5]);
            p[kk * 4 + 1] = pack_e4(e[b0 + 2], e[b0 + 3], e[b0 + 6], e[b0 + 7]);
            p[kk * 4 + 2] = pack_e4(e[b0 + 8], e[b0 + 9], e[b0 + 12], e[b0 + 13]);
            p[kk * 4 + 3] = pack_e4(e[b0 + 10], e[b0 + 11], e[b0 + 14], e[b0 + 15]);
        }
        l[0] = l[0] * alpha[0] + l0; l[1] = l[1] * alpha[1] + l1;
        return;
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        float a0, a1, b0, b1;
        if (j < NPOLY) {
            a0 = ex2p(fmaf(s[j * 4 + 0], scale, -mn0)); a1 = ex2p(fmaf(s[j * 4 + 1], scale, -mn0));
            b0 = ex2p(fmaf(s[j * 4 + 2], scale, -mn1)); b1 = ex2p(fmaf(s[j * 4 + 3], scale, -mn1));
        } else {
            a0 = ex2(fmaf(s[j * 4 + 0], scale, -mn0)); a1 = ex2(fmaf(s[j * 4 + 1], scale, -mn0));
            b0 = ex2(fmaf(s[j * 4 + 2], scale, -mn1)); b1 = ex2(fmaf(s[j * 4 + 3], scale, -mn1));
        }
        if (!ONES) { l0 += a0 + a1; l1 += b0 + b1; }
        // A fragment for k16 step kk = j/2: regs (kk*4 + 0/1) from n8 block 2kk, (kk*4 + 2/3) from 2kk+1.
        p[(j >> 1) * 4 + (j & 1) * 2 + 0] = pack_h2(a0, a1);
        p[(j >> 1) * 4 + (j & 1) * 2 + 1] = pack_h2(b0, b1);
    }
    l[0] = l[0] * alpha[0] + l0; l[1] = l[1] * alpha[1] + l1;
}
__device__ __forceinline__ void rescale(float (&o)[32], const float (&alpha)[2], float (&ls)[4]) {
    if (LAZY && __all_sync(0xffffffff, alpha[0] == 1.f && alpha[1] == 1.f)) return;
    if (ONES) { ls[0] *= alpha[0]; ls[1] *= alpha[0]; ls[2] *= alpha[1]; ls[3] *= alpha[1]; }
#pragma unroll
    for (int j = 0; j < 8; ++j) { o[j * 4 + 0] *= alpha[0]; o[j * 4 + 1] *= alpha[0]; o[j * 4 + 2] *= alpha[1]; o[j * 4 + 3] *= alpha[1]; }
}

#define VADDR2(x, hh) (PV8 ? smem_u32(&sm.vt[x][hh][0]) : smem_u32(&sm.v[x][hh][0]))
template <int NS, int PIPE, int MINB>
__global__ void __launch_bounds__(256, MINB) cmicro(const uint8_t* __restrict__ kg, const __half* __restrict__ vg,
                                                    const uint8_t* __restrict__ qg, float* __restrict__ out, int nt,
                                                    float scale, const uint8_t* __restrict__ vtg) {
    extern __shared__ __align__(1024) uint8_t smraw[];
    Smem<NS>& sm = *reinterpret_cast<Smem<NS>*>((reinterpret_cast<uintptr_t>(smraw) + 1023) & ~uintptr_t(1023));
    int tid = threadIdx.x;
    // Preload: kg [NS][BN][128] fp8, vg [NS][2][BN][HD] fp16, qg [2][MR][HD] fp8.
    for (int i = tid; i < NS * BN * 128 / 16; i += 256) {
        int s = i / (BN * 8), o = (i % (BN * 8)) * 16;
        *reinterpret_cast<uint4*>(&sm.k[s][swz128(o)]) = reinterpret_cast<const uint4*>(kg)[i];
    }
    for (int i = tid; i < NS * 2 * BN * HD / 8; i += 256) {
        int sh = i / (BN * 8), o = (i % (BN * 8)) * 16;
        *reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(&sm.v[sh / 2][sh % 2][0]) + swz128(o)) =
            reinterpret_cast<const uint4*>(vg)[i];
    }
    for (int i = tid; i < 2 * MR * HD / 16; i += 256) {
        int h = i / (MR * 4), o = (i % (MR * 4)) * 16;
        *reinterpret_cast<uint4*>(&sm.q[h][swz64(o)]) = reinterpret_cast<const uint4*>(qg)[i];
    }
    for (int i = tid; i < 2048; i += 256) sm.ones[i] = __float2half(1.f);
    for (int i = tid; i < NS * 2 * HD * BN / 16; i += 256) {
        int sh = i / (HD * 4), o = (i % (HD * 4)) * 16;
        *reinterpret_cast<uint4*>(&sm.vt[sh / 2][sh % 2][swz64(o)]) = reinterpret_cast<const uint4*>(vtg)[i];
    }
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    __syncthreads();
    int h = tid / 128, lane = tid % 32, w = (tid % 128) / 32;
    uint32_t qa = smem_u32(&sm.q[h][0]);
    float o[32], s[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) o[i] = 0.f;
    float m[2] = {-1e30f, -1e30f}, l[2] = {0.f, 0.f}, alpha[2];
    uint32_t pa[16], pb[16];
    float ls[4] = {0.f, 0.f, 0.f, 0.f};
    uint32_t onea = smem_u32(&sm.ones[0]);
    if (!PIPE) {
        for (int i = 0; i < nt; ++i) {
            int sl = i % NS;
            issue_qk(s, qa, smem_u32(&sm.k[sl][0]) + h * 64);
            wg_wait<0>();
            fence_regs(s);
            softmax(s, m, l, pa, alpha, scale);
            rescale(o, alpha, ls);
            fence_regs(o);
            issue_pv(o, pa, VADDR2(sl, h), ls, onea);
            wg_wait<0>();
            fence_regs(o);
        }
    } else {
        // FA3 intra-warpgroup pipelining: P@V(i-1) runs during softmax(i).
        issue_qk(s, qa, smem_u32(&sm.k[0][0]) + h * 64);
        wg_wait<0>();
        fence_regs(s);
        softmax(s, m, l, pa, alpha, scale);
        int i = 1;
        for (; i + 1 < nt; i += 2) {
            issue_qk(s, qa, smem_u32(&sm.k[i % NS][0]) + h * 64);
            fence_regs(pa);
            issue_pv(o, pa, VADDR2((i - 1) % NS, h), ls, onea);
            wg_wait<1>();
            fence_regs(s);
            softmax(s, m, l, pb, alpha, scale);
            wg_wait<0>();
            fence_regs(o); fence_regs(ls);
            rescale(o, alpha, ls);
            fence_regs(o);
            issue_qk(s, qa, smem_u32(&sm.k[(i + 1) % NS][0]) + h * 64);
            fence_regs(pb);
            issue_pv(o, pb, VADDR2(i % NS, h), ls, onea);
            wg_wait<1>();
            fence_regs(s);
            softmax(s, m, l, pa, alpha, scale);
            wg_wait<0>();
            fence_regs(o); fence_regs(ls);
            rescale(o, alpha, ls);
            fence_regs(o);
        }
        if (i < nt) {
            issue_qk(s, qa, smem_u32(&sm.k[i % NS][0]) + h * 64);
            fence_regs(pa);
            issue_pv(o, pa, VADDR2((i - 1) % NS, h), ls, onea);
            wg_wait<1>();
            fence_regs(s);
            softmax(s, m, l, pb, alpha, scale);
            wg_wait<0>();
            fence_regs(o); fence_regs(ls);
            rescale(o, alpha, ls);
            fence_regs(o);
            issue_pv(o, pb, VADDR2(i % NS, h), ls, onea);
        } else {
            issue_pv(o, pa, VADDR2((i - 1) % NS, h), ls, onea);
        }
        wg_wait<0>();
        fence_regs(o);
    }
    if (ONES) { l[0] = ls[0]; l[1] = ls[2]; }
    // quad-reduce l
    if (!ONES) {
    l[0] += __shfl_xor_sync(0xffffffff, l[0], 1); l[0] += __shfl_xor_sync(0xffffffff, l[0], 2);
    l[1] += __shfl_xor_sync(0xffffffff, l[1], 1); l[1] += __shfl_xor_sync(0xffffffff, l[1], 2);
    }
    if (blockIdx.x == 0) {
        // out [2][MR][HD + 2]: acc, m, l
        int r0 = w * 16 + lane / 4;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            int c = j * 8 + (lane % 4) * 2;
            float* p0 = out + (h * MR + r0) * (HD + 2);
            float* p1 = out + (h * MR + r0 + 8) * (HD + 2);
            p0[c] = o[j * 4 + 0]; p0[c + 1] = o[j * 4 + 1]; p1[c] = o[j * 4 + 2]; p1[c + 1] = o[j * 4 + 3];
        }
        if (lane % 4 == 0) {
            out[(h * MR + r0) * (HD + 2) + HD] = m[0]; out[(h * MR + r0) * (HD + 2) + HD + 1] = l[0];
            out[(h * MR + r0 + 8) * (HD + 2) + HD] = m[1]; out[(h * MR + r0 + 8) * (HD + 2) + HD + 1] = l[1];
        }
    } else if (o[0] == 12345.f && o[5] == 1.f) {
        out[0] = l[0];  // keep the work live
    }
}


template <int NS, int PIPE, int MINB>
void launch(torch::Tensor kg, torch::Tensor vg, torch::Tensor qg, torch::Tensor out, int nt, float scale, int grid, torch::Tensor vt) {
    auto f = cmicro<NS, PIPE, MINB>;
    int smem = sizeof(Smem<NS>) + 1024;
    cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    f<<<grid, 256, smem, c10::cuda::getCurrentCUDAStream()>>>(kg.data_ptr<uint8_t>(), (const __half*)vg.data_ptr(),
        qg.data_ptr<uint8_t>(), out.data_ptr<float>(), nt, scale, vt.data_ptr<uint8_t>());
}
void run(int variant, torch::Tensor kg, torch::Tensor vg, torch::Tensor qg, torch::Tensor out, int nt, float scale, int grid, torch::Tensor vt) {
    switch (variant) {
        case 0: launch<3, 0, 2>(kg, vg, qg, out, nt, scale, grid, vt); break;
        case 1: launch<3, 1, 2>(kg, vg, qg, out, nt, scale, grid, vt); break;
        case 2: launch<3, 1, 1>(kg, vg, qg, out, nt, scale, grid, vt); break;
        case 3: launch<2, 1, 2>(kg, vg, qg, out, nt, scale, grid, vt); break;
        case 4: launch<2, 0, 3>(kg, vg, qg, out, nt, scale, grid, vt); break;
        case 5: launch<2, 1, 3>(kg, vg, qg, out, nt, scale, grid, vt); break;
    }
}

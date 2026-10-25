// Native verify-full attention (L 8, G 8, D 64, HKV 2) = _gw_verify gbal with LZ 10, in CUDA (session 111).
// v5: Gluon's layout (one CTA of 12 warps per program, 2 per SM; consumers 104 registers): warps 0-1 gather (rows
// 32w + ..), warps 2-3 widen (units split by warp). (v4: one CTA of 20 warps per SM running programs 2c and 2c + 1 (an SM sub-partition's 16K registers can't hold two
// 9-warp CTAs at 112): warpgroup 0 (32 registers): warp q gathers program 2c + q's K/V rows with cp.async whose
// completion the slot's mbarrier tracks (no waits on its own copies), warp 2 + q widens its V to FP16 [key][d] in a
// ring of 2; warpgroups 1-4 (112 registers via setmaxnreg; one per (program, KV head)) run QK^T (FP8 wgmma), LZ 10's fixed-reference softmax (first NV_PK tiles exact
// against row max + 14 binades, later ones against that fixed reference; overflowing rows -> m = +inf -> REDO in
// _combine_bal) and P@V with FP16 P (RS wgmma), FA3 intra-warpgroup pipelined: QK(i) and PV(i-1) in flight while
// softmax(i) runs. Partials [HKV, NID, 64, 64] at id p + b, as _gw_verify.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <type_traits>
#include <stdint.h>

namespace nv {
constexpr int BN = 64, HD = 64, MR = 64;
#ifndef NV_ABL
#define NV_ABL 0
#endif
#ifndef NV_LAG
#define NV_LAG 1
#endif
#ifndef NV_PFD
#define NV_PFD 4
#endif
#ifndef NV_PK
#define NV_PK 4  // exact tiles per segment (LZ_PK)
#endif
#ifndef NV_DELTA
#define NV_DELTA 14.f  // FP16 headroom of the fixed reference (LZ_DELTA, binades)
#endif
#ifndef NV_ABL
#define NV_ABL 0
#endif
#ifndef NV_UNR
#define NV_UNR 2
#endif
#ifndef NV_MINB
#define NV_MINB 2
#endif
#ifndef NV_PIPE
#define NV_PIPE 1
#endif

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
__device__ __forceinline__ void qk_k32(float (&s)[32], uint64_t da, uint64_t db, int acc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 " R32 ", %32, %33, p, 1, 1;\n}\n"
                 : D32(s) : "l"(da), "l"(db), "r"(acc));
}
// O[64 x 64] += P[64 x 16 fp16, registers] V[16 x 64 fp16, MN-major smem], one k16 step.
__device__ __forceinline__ void pv_k16(float (&o)[32], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 " R32 ", {%32,%33,%34,%35}, %36, 1, 1, 1, 1;\n"
                 : D32(o) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
}
__device__ __forceinline__ uint32_t pack_h2(float a, float b) {
    uint32_t d; asm("cvt.rn.f16x2.f32 %0, %2, %1;" : "=r"(d) : "f"(a), "f"(b)); return d;
}
__device__ __forceinline__ uint32_t pack_e4(float a, float b, float c, float d) {
    uint16_t lo, hi;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(lo) : "f"(a), "f"(b));
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(hi) : "f"(c), "f"(d));
    return (uint32_t)lo | ((uint32_t)hi << 16);
}
// Two E4M3 bytes (low byte first) -> FP16x2 (exact).
__device__ __forceinline__ uint32_t e4x2_h2(uint32_t w16) {
    uint32_t d; asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(d) : "h"((uint16_t)w16)); return d;
}
__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) {
    uint32_t d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d;
}
__device__ __forceinline__ void mb_init(uint64_t* b, int n) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(b)), "r"(n) : "memory");
}
__device__ __forceinline__ void mb_arrive(uint64_t* b) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void mb_wait(uint64_t* b, uint32_t par) {
    asm volatile("{\n.reg .pred p;\nW: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1;\n@!p bra W;\n}\n"
                 ::"r"(smem_u32(b)), "r"(par) : "memory");
}
__device__ __forceinline__ void cp16(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp16ca(uint32_t dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }


#ifndef NV_WAB
#define NV_WAB 0
#endif
#ifndef NV_CVP
#define NV_CVP 2  // where step i widens V(i): 0 before QK(i), 1 between QK(i) and PV(i-1), 2 after both
#endif
__device__ __forceinline__ void cp_mbar_arrive(uint64_t* b) {
    asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(smem_u32(b)) : "memory");
}
constexpr int UNR = NV_UNR;
template <int NS> struct Smem {
    alignas(1024) uint8_t k[NS][BN * 128];    // both heads' K rows, SW128 (wgmma B, K-major)
    alignas(1024) uint8_t vs[NS][BN * 128];   // V rows as gathered; 16-byte chunk c of row r at c ^ (r & 7)
    alignas(1024) __half vh[2][2][BN * HD];   // FP16 V [key][d] per head, SW128 (wgmma B, MN-major), ring of 2
    alignas(1024) uint8_t q[2][MR * HD];      // per head FP8 queries [row][d], SW64
    uint64_t ready[NS], empty[NS];            // K/V slot landed (32 cp.async arrivals) / done (8 QK warps + widen)
    uint64_t vfull[2], vempty[2];             // FP16 V slot widened / consumed (8 P@V warps)
};

struct Range { int start, end, b, excl; };
// _gw_bal: program p's tiles [start, end) of the batch's tiles end to end; its first request b and that request's
// first tile excl.
__device__ __forceinline__ Range bal(const int* __restrict__ seq, int B, int p, int P) {
    int tt = 0;
    for (int b = 0; b < B; ++b) tt += (seq[b] + BN - 1) / BN;
    tt = max(tt, 1);
    Range r;
    r.start = (int)((long long)p * tt / P);
    r.end = (int)((long long)(p + 1) * tt / P);
    int incl = 0;
    r.b = 0; r.excl = 0;
    for (int b = 0; b < B; ++b) {
        int t = (seq[b] + BN - 1) / BN;
        if (incl + t <= r.start) { r.b = b + 1; r.excl = incl + t; }
        incl += t;
    }
    return r;
}

// Gather warp pair: tile i's 64 K and V rows into slot i % NS; warp w of the pair takes rows 32w + lane / 8 + 4j,
// 8 lanes per 128-byte row. Row indices come two tiles ahead from plain loads (lane holds row 32w + lane) and are
// spread by shuffles.
template <int NS>
__device__ __forceinline__ void producer(Smem<NS>& sm, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
                                         const int* __restrict__ pt, const int* __restrict__ seq, Range rg, int B,
                                         int maxp, int page, int w) {
    const int lane = threadIdx.x & 31;
    const int total = max(rg.end - rg.start, 0);
    int b = rg.b, excl = rg.excl;
    int sb = (b < B) ? seq[b] : 0;
    int incl = excl + (sb + BN - 1) / BN;
    auto rows = [&](int j) -> int {
        if (j >= total) return 0;
        const int gi = rg.start + j;
        while (gi >= incl && b + 1 < B) { ++b; excl = incl; sb = seq[b]; incl = excl + (sb + BN - 1) / BN; }
        const int t = max(min((gi - excl) * BN + 32 * w + lane, sb - 1), 0);
        const int* ptr = pt + (long long)b * maxp;
        return page == 1 ? ptr[t] : ptr[t / page] * page + t % page;
    };
    int c = rows(0), n = rows(1);
    const int ch = lane & 7, r0 = 32 * w + (lane >> 3);
    const uint32_t kb0 = smem_u32(&sm.k[0][0]), vb0 = smem_u32(&sm.vs[0][0]);
    for (int i = 0; i < (NV_ABL == 9 ? 0 : NV_ABL == 8 ? min(total, NS) : total); ++i) {
        const int s = i % NS;
        mb_wait(&sm.empty[s], ((i / NS) & 1) ^ 1);
        if (NV_ABL != 6 || i < NS) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int r = r0 + 4 * j;
                const int e = __shfl_sync(0xffffffff, c, (lane >> 3) + 4 * j);
                const long long off = (long long)e * 128 + ch * 16;
                const uint32_t d = s * (BN * 128) + r * 128 + ((ch ^ (r & 7)) << 4);
                cp16(kb0 + d, kc + off);
                cp16(vb0 + d, vc + off);
            }
        }
        cp_mbar_arrive(&sm.ready[s]);
        c = n;
        n = rows(i + 2);
    }
    asm volatile("cp.async.wait_all;" ::: "memory");
}

// Widen warp: tile t's V (slot t % NS) -> FP16 slot t & 1 once P@V(t - 2) has drained it. Unit u = lane + 32k
// (k 0..15): row (u & 7) + 8 (u >> 6), chunk (u >> 3) & 7; 8 consecutive lanes take 8 rows of one chunk.
template <int NS>
__device__ __forceinline__ void widener(Smem<NS>& sm, int total, int w) {
    const int lane = threadIdx.x & 31;
    for (int t = 0; t < total; ++t) {
        const int s = t % NS, sv = t & 1;
        mb_wait(&sm.ready[s], (t / NS) & 1);
        mb_wait(&sm.vempty[sv], ((t >> 1) & 1) ^ 1);
        const uint8_t* vsb = &sm.vs[s][0];
        if (NV_ABL != 7 || t < 2)
#pragma unroll 4
        for (int k = 0; k < 8; ++k) {
            const int u = lane + 32 * (2 * k + w), r = (u & 7) + 8 * (u >> 6), cc = (u >> 3) & 7;
            const uint4 x = *reinterpret_cast<const uint4*>(vsb + r * 128 + ((cc ^ (r & 7)) << 4));
            uint8_t* vhb = reinterpret_cast<uint8_t*>(&sm.vh[sv][cc >> 2][0]);
            const uint32_t o = r * 128 + (cc & 3) * 32;
            *reinterpret_cast<uint4*>(vhb + swz128(o)) = make_uint4(e4x2_h2(x.x), e4x2_h2(x.x >> 16), e4x2_h2(x.y), e4x2_h2(x.y >> 16));
            *reinterpret_cast<uint4*>(vhb + swz128(o + 16)) = make_uint4(e4x2_h2(x.z), e4x2_h2(x.z >> 16), e4x2_h2(x.w), e4x2_h2(x.w >> 16));
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncwarp();
        if (lane == 0) { mb_arrive(&sm.vfull[sv]); mb_arrive(&sm.empty[s]); }  // 2 widen warps
    }
}

__device__ __forceinline__ void issue_qk(float (&s)[32], uint32_t qa, uint32_t ka) {
    wg_fence();
    qk_k32(s, mdesc(qa, 512, 2), mdesc(ka, 1024, 1), 0);
    qk_k32(s, mdesc(qa + 32, 512, 2), mdesc(ka + 32, 1024, 1), 1);
    wg_commit();
}
// P (FP16 A fragments: k16 step kk = regs 4kk .. 4kk + 3 = (row r0, n8 block 2kk), (r1, 2kk), (r0, 2kk+1),
// (r1, 2kk+1)) times this head's V [key][d] (MN-major, SW128).
__device__ __forceinline__ void issue_pv(float (&o)[32], const uint32_t (&p)[16], uint32_t va) {
    wg_fence();
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) pv_k16(o, p[4 * kk], p[4 * kk + 1], p[4 * kk + 2], p[4 * kk + 3], mdesc(va + kk * 2048, 1024, 1));
    wg_commit();
}
__device__ __forceinline__ void mask(float (&s)[32], int lane, int nvalid) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int col = j * 8 + (lane & 3) * 2;
        if (col >= nvalid) { s[j * 4 + 0] = -INFINITY; s[j * 4 + 2] = -INFINITY; }
        if (col + 1 >= nvalid) { s[j * 4 + 1] = -INFINITY; s[j * 4 + 3] = -INFINITY; }
    }
}
// Softmax of one tile in place (s -> P in FP32). EXACT: reference m = max(m, row max * scale + NV_DELTA), returns
// alpha; else against the fixed m. l = per-thread partial row sums (quad-reduced at the end).
template <bool EXACT>
__device__ __forceinline__ void softmax(float (&s)[32], float (&m)[2], float (&l)[2], float (&alpha)[2], float scale) {
    if (EXACT) {
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
        const float mn0 = fmaxf(m[0], fmaf(mx0, scale, NV_DELTA)), mn1 = fmaxf(m[1], fmaf(mx1, scale, NV_DELTA));
        alpha[0] = ex2(m[0] - mn0); alpha[1] = ex2(m[1] - mn1);
        m[0] = mn0; m[1] = mn1;
    }
    const float o0 = m[0], o1 = m[1];
    float l0 = 0.f, l1 = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        s[j * 4 + 0] = ex2(fmaf(s[j * 4 + 0], scale, -o0)); s[j * 4 + 1] = ex2(fmaf(s[j * 4 + 1], scale, -o0));
        s[j * 4 + 2] = ex2(fmaf(s[j * 4 + 2], scale, -o1)); s[j * 4 + 3] = ex2(fmaf(s[j * 4 + 3], scale, -o1));
        l0 += s[j * 4 + 0] + s[j * 4 + 1]; l1 += s[j * 4 + 2] + s[j * 4 + 3];
    }
    if (EXACT) { l[0] = l[0] * alpha[0] + l0; l[1] = l[1] * alpha[1] + l1; }
    else { l[0] += l0; l[1] += l1; }
}
__device__ __forceinline__ void pack(const float (&s)[32], uint32_t (&p)[16]) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        p[(j >> 1) * 4 + (j & 1) * 2 + 0] = pack_h2(s[j * 4 + 0], s[j * 4 + 1]);
        p[(j >> 1) * 4 + (j & 1) * 2 + 1] = pack_h2(s[j * 4 + 2], s[j * 4 + 3]);
    }
}
__device__ __forceinline__ void rescale(float (&o)[32], const float (&alpha)[2]) {
#pragma unroll
    for (int j = 0; j < 8; ++j) { o[j * 4 + 0] *= alpha[0]; o[j * 4 + 1] *= alpha[0]; o[j * 4 + 2] *= alpha[1]; o[j * 4 + 3] *= alpha[1]; }
}
template <int NS>
__device__ __forceinline__ void consumer(Smem<NS>& sm, int h, const __nv_bfloat16* __restrict__ qkv,
                                         const int* __restrict__ seq, Range rg, int B, int p, int nid, int qkv_w,
                                         float scale, float* __restrict__ pacc, float* __restrict__ pml, int bar) {
    const int tid = threadIdx.x & 127, lane = tid & 31, w = tid >> 5;
    const uint32_t qa = smem_u32(&sm.q[h][0]);
    int b = rg.b, excl = rg.excl, it = 0;
    int gi = rg.start;
    const int end = rg.end;
    if (end <= rg.start) return;
    while (true) {
        if (b >= B) break;
        const int sb = seq[b];
        const int nb = (sb + BN - 1) / BN;
        const int t0 = max(gi, excl) - excl, t1 = min(end, excl + nb) - excl;
        if (excl >= end) break;
        const int n = max(t1 - t0, 0), lo = t0 * BN, hi = min(t1 * BN, sb);
        {   // Queries: row r = token r / 8, head h * 8 + r % 8; 2 threads per row (32 d each); RMS-norm, FP8.
            const int r = tid >> 1, half = tid & 1;
            const __nv_bfloat16* src = qkv + (long long)(b * 8 + (r >> 3)) * qkv_w + (h * 8 + (r & 7)) * HD + half * 32;
            uint4 raw[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) raw[j] = reinterpret_cast<const uint4*>(src)[j];
            float x[32], ss = 0.f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const __nv_bfloat162* hp = reinterpret_cast<const __nv_bfloat162*>(&raw[j]);
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    float2 f = __bfloat1622float2(hp[k]);
                    x[j * 8 + 2 * k] = f.x; x[j * 8 + 2 * k + 1] = f.y;
                    ss += f.x * f.x + f.y * f.y;
                }
            }
            ss += __shfl_xor_sync(0xffffffff, ss, 1);
            const float rs = rsqrtf(ss / HD + 1e-5f);
            uint32_t pk[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) pk[j] = pack_e4(x[4 * j] * rs, x[4 * j + 1] * rs, x[4 * j + 2] * rs, x[4 * j + 3] * rs);
            // The previous segment's QK^T have completed (wait 0), so q can be overwritten.
            asm volatile("bar.sync %0, 128;" ::"r"(bar) : "memory");
            uint8_t* qd = &sm.q[h][0];
            *reinterpret_cast<uint4*>(qd + swz64(r * 64 + half * 32)) = make_uint4(pk[0], pk[1], pk[2], pk[3]);
            *reinterpret_cast<uint4*>(qd + swz64(r * 64 + half * 32 + 16)) = make_uint4(pk[4], pk[5], pk[6], pk[7]);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("bar.sync %0, 128;" ::"r"(bar) : "memory");
        }
        float o[32], s[32];
#pragma unroll
        for (int i = 0; i < 32; ++i) o[i] = 0.f;
        float m[2] = {-1e30f, -1e30f}, l[2] = {0.f, 0.f}, alpha[2];
        uint32_t pp[16];
        if (NV_ABL == 9) {
        } else if (NV_ABL == 3) {
            for (int i = 0; i < n; ++i) {
                const int t = it + i;
                mb_wait(&sm.ready[t % NS], (t / NS) & 1); if (lane == 0) mb_arrive(&sm.empty[t % NS]);
                mb_wait(&sm.vfull[t & 1], (t >> 1) & 1); if (lane == 0) mb_arrive(&sm.vempty[t & 1]);
            }
        } else if (n > 0) {
            {
                const int sl = it % NS;
                mb_wait(&sm.ready[sl], (it / NS) & 1);
                issue_qk(s, qa, smem_u32(&sm.k[sl][0]) + h * 64);
                wg_wait<0>();
                fence_regs(s);
                if (lane == 0) mb_arrive(&sm.empty[sl]);
                if (hi - lo < BN) mask(s, lane, hi - lo);
                softmax<true>(s, m, l, alpha, scale);
                pack(s, pp);
            }
            // Tile i: QK(i) and PV(i - 1) in flight, softmax(i) under PV(i - 1); P is repacked after PV(i - 1) has
            // drained (one P buffer). K/V slots are released after QK^T, FP16 V slots after P@V.
            auto step = [&](auto exact, int i) {
                constexpr bool EX = decltype(exact)::value;
                const int tc = it + i, tp = tc - 1, sc = tc % NS, vp = tp & 1;
                mb_wait(&sm.ready[sc], (tc / NS) & 1);
                issue_qk(s, qa, smem_u32(&sm.k[sc][0]) + h * 64);
                mb_wait(&sm.vfull[vp], (tp >> 1) & 1);
                fence_regs(pp);
                issue_pv(o, pp, smem_u32(&sm.vh[vp][h][0]));
                if (NV_PIPE) wg_wait<1>(); else wg_wait<0>();
                fence_regs(s);
                if (lane == 0) mb_arrive(&sm.empty[sc]);
                const int nv = hi - lo - i * BN;
                if (nv < BN) mask(s, lane, nv);
                softmax<EX>(s, m, l, alpha, scale);
                wg_wait<0>();
                fence_regs(o);
                fence_regs(pp);
                if (lane == 0) mb_arrive(&sm.vempty[vp]);
                if (EX) { rescale(o, alpha); fence_regs(o); }
                pack(s, pp);
            };
            const int nx = min(n, NV_PK);
            int i = 1;
            if (NV_ABL != 1) for (; i < nx; ++i) step(std::true_type{}, i);
#pragma unroll UNR
            for (; i < n; ++i) step(std::false_type{}, i);
            {
                const int tp = it + n - 1, vp = tp & 1;
                mb_wait(&sm.vfull[vp], (tp >> 1) & 1);
                fence_regs(pp);
                issue_pv(o, pp, smem_u32(&sm.vh[vp][h][0]));
                wg_wait<0>();
                fence_regs(o);
                if (lane == 0) mb_arrive(&sm.vempty[vp]);
            }
        }
        it += n;
        l[0] += __shfl_xor_sync(0xffffffff, l[0], 1); l[0] += __shfl_xor_sync(0xffffffff, l[0], 2);
        l[1] += __shfl_xor_sync(0xffffffff, l[1], 1); l[1] += __shfl_xor_sync(0xffffffff, l[1], 2);
        // A row whose scores outgrew the fixed reference has a non-finite acc: store m = +inf (REDO in the merge).
        float a0 = 0.f, a1 = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) { a0 += o[j * 4 + 0] + o[j * 4 + 1]; a1 += o[j * 4 + 2] + o[j * 4 + 3]; }
        a0 += __shfl_xor_sync(0xffffffff, a0, 1); a0 += __shfl_xor_sync(0xffffffff, a0, 2);
        a1 += __shfl_xor_sync(0xffffffff, a1, 1); a1 += __shfl_xor_sync(0xffffffff, a1, 2);
        const float m0 = (a0 - a0 == 0.f) ? m[0] : INFINITY, m1 = (a1 - a1 == 0.f) ? m[1] : INFINITY;
        // Partial id p + b, rows w * 16 + lane / 4 (+ 8).
        const long long part = ((long long)h * nid + p + b) * MR;
        const int ra = w * 16 + (lane >> 2);
        float* pa0 = pacc + (part + ra) * HD;
        float* pa1 = pa0 + 8 * HD;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int col = j * 8 + (lane & 3) * 2;
            *reinterpret_cast<float2*>(pa0 + col) = make_float2(o[j * 4 + 0], o[j * 4 + 1]);
            *reinterpret_cast<float2*>(pa1 + col) = make_float2(o[j * 4 + 2], o[j * 4 + 3]);
        }
        if ((lane & 3) == 0) {
            *reinterpret_cast<float2*>(pml + (part + ra) * 2) = make_float2(m0, l[0]);
            *reinterpret_cast<float2*>(pml + (part + ra + 8) * 2) = make_float2(m1, l[1]);
        }
        gi = excl + t1;
        excl += nb;
        ++b;
        if (excl >= end) break;
    }
}

template <int NS>
__global__ void __launch_bounds__(384, 2) nvf(const __nv_bfloat16* __restrict__ qkv, const uint8_t* __restrict__ kc,
                                              const uint8_t* __restrict__ vc, const float* __restrict__ ks,
                                              const int* __restrict__ pt, const int* __restrict__ seq,
                                              float* __restrict__ pacc, float* __restrict__ pml, float qk_scale,
                                              int B, int P, int nid, int maxp, int page, int qkv_w) {
    extern __shared__ __align__(1024) uint8_t smraw[];
    Smem<NS>& sm = *reinterpret_cast<Smem<NS>*>((reinterpret_cast<uintptr_t>(smraw) + 1023) & ~uintptr_t(1023));
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    const int tid = threadIdx.x, wg = __shfl_sync(0xffffffff, tid >> 7, 0);
    if (tid == 0) {
        for (int i = 0; i < NS; ++i) { mb_init(&sm.ready[i], 64); mb_init(&sm.empty[i], 10); }
        for (int i = 0; i < 2; ++i) { mb_init(&sm.vfull[i], 2); mb_init(&sm.vempty[i], 8); }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    const int p = blockIdx.x;
    const Range rg = bal(seq, B, p, P);
    if (wg == 0) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(NV_PREGS));
        if (tid < 64) producer<NS>(sm, kc, vc, pt, seq, rg, B, maxp, page, tid >> 5);
        else widener<NS>(sm, max(rg.end - rg.start, 0), (tid >> 5) & 1);
    } else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(NV_CREGS));
        consumer<NS>(sm, wg - 1, qkv, seq, rg, B, p, nid, qkv_w, qk_scale * ks[0], pacc, pml, 1 + wg);
    }
    asm volatile("griddepcontrol.wait;" ::: "memory");
}
}  // namespace nv

void nvf(torch::Tensor qkv, torch::Tensor kc, torch::Tensor vc, torch::Tensor ks, torch::Tensor pt,
         torch::Tensor seq, torch::Tensor pacc, torch::Tensor pml, double qk_scale, int64_t P, int64_t nid,
         int64_t page) {
    auto f = nv::nvf<NV_NS>;
    const int smem = (int)sizeof(nv::Smem<NV_NS>) + 1024;
    static bool once = false;
    if (!once) { cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); once = true; }
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)P);
    cfg.blockDim = dim3(384);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = c10::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, f, (const __nv_bfloat16*)qkv.data_ptr(), (const uint8_t*)kc.data_ptr(),
                       (const uint8_t*)vc.data_ptr(), (const float*)ks.data_ptr(), (const int*)pt.data_ptr(),
                       (const int*)seq.data_ptr(), (float*)pacc.data_ptr(), (float*)pml.data_ptr(), (float)qk_scale,
                       (int)seq.size(0), (int)P, (int)nid, (int)pt.size(1), (int)page, (int)qkv.size(-1));
}
int64_t nvf_occ() {
    auto f = nv::nvf<NV_NS>;
    const int smem = (int)sizeof(nv::Smem<NV_NS>) + 1024;
    cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    int n = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, f, 384, smem);
    return n * 1000000 + smem;
}

// Native verify-full attention (L 8, G 8, D 64, HKV 2): _gw_verify's balanced-split (gbal) program in CUDA, with a
// lazy-max online softmax. One CTA per program p (2 per SM): warpgroup 0 gathers K/V rows with cp.async and
// converts V to FP16 (as _gw_verify), warpgroups 1/2 (one per KV head) run QK^T (FP8 wgmma), the softmax and P@V
// with P in FP16 (V FP16 [key][d], MN-major smem B). Lazy max: exponents against the running maxima without a
// per-tile row max; a warp whose partial row sums all stay <= 2^8 (so every P <= 2^8, P >= 0) keeps m, l and the
// accumulators as they are (exact online softmax with a stale offset); otherwise (first tile, a row grew by > 8,
// inf/NaN) the warp redoes the tile with the usual max update. Partials [HKV, NID, 64, 64] at id p + b.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
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
#ifndef NV_TR
#define NV_TR 1
#endif
#ifndef NV_CV
#define NV_CV 0
#endif
#ifndef NV_TH
#define NV_TH 256.f
#endif
#ifndef NV_LAZY
#define NV_LAZY 1
#endif

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ uint64_t mdesc(uint32_t addr, uint32_t sbo, uint32_t mode) {
    return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)(sbo >> 4) << 32) | ((uint64_t)mode << 62);
}
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
__device__ __forceinline__ void pv8_k32(float (&o)[32], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 " R32 ", {%32,%33,%34,%35}, %36, 1, 1, 1;\n"
                 : D32(o) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
}
__device__ __forceinline__ uint32_t pack_e4(float a, float b, float c, float d) {
    uint16_t lo, hi;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(lo) : "f"(a), "f"(b));
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(hi) : "f"(c), "f"(d));
    return (uint32_t)lo | ((uint32_t)hi << 16);
}
__device__ __forceinline__ uint32_t cvt8(uint32_t x) {  // the low two E4M3 bytes of x -> f16x2
    uint32_t d; asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(d) : "h"((uint16_t)(x & 0xFFFF))); return d;
}
__device__ __forceinline__ uint32_t pack_h2(float a, float b) {
    __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<uint32_t*>(&h);
}
__device__ __forceinline__ void pv_k16(float (&o)[32], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 " R32 ", {%32,%33,%34,%35}, %36, 1, 1, 1, 1;\n"
                 : D32(o) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
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

template <int NS> struct Smem {
    alignas(1024) uint8_t k[NS][BN * 128];     // both heads' K rows, SW128 (wgmma B, K-major)
    alignas(1024) uint8_t vs[NS][BN * 128];    // V rows as gathered; 16-byte chunk c of row r at c ^ vsw(r)
    alignas(1024) __half vh[NS][2][BN * HD];    // per head FP16 V [key][d], SW128 (wgmma B, MN-major)
    alignas(1024) uint8_t q[2][MR * HD];       // per head FP8 queries [row][d], SW64
    int ptr[8][BN];                            // page 1: page-table entries of tiles i .. i + NV_PFD (ring of 8)
    int lim[8];                                // and each tile's last valid key
    uint64_t ready[NS], empty[NS];
};
// Staging swizzle: SW128 (r & 7) for the ldmatrix transpose (8 consecutive rows per phase), else conflict-free for
// the 4-row register transpose.
__device__ __forceinline__ uint32_t vsw(int r) { return NV_TR ? (r & 7) : (((r >> 1) & 3) | (((r >> 4) & 1) << 2)); }

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

template <int NS, bool PAGE1>
__device__ __forceinline__ void producer(Smem<NS>& sm, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
                                         const int* __restrict__ pt, const int* __restrict__ seq, Range rg, int B,
                                         int maxp, int page) {
    const int tid = threadIdx.x;  // 0..127
    // Gathers: thread copies row tid / 2, 16-byte chunks 4 (tid & 1) + 0..3 of K and V.
    const int prow = tid >> 1, pc0 = (tid & 1) * 4;
    // Transpose unit: keys 16g + {2t, 2t+1, 2t+8, 2t+9}, 16 bytes of chunk c (head c >> 2, d 16 (c & 3) + 0..15).
    const int lane = tid & 31, t4 = lane & 3, g = (lane >> 2) & 3, c = ((tid >> 5) << 1) | (lane >> 4);
    const int r0 = 16 * g + 2 * t4;
    const uint32_t vsoff = (uint32_t)(r0 * 128) + ((c ^ vsw(r0)) << 4);
    // V^T word of d = 16 (c & 3) + 4w + jj at head base + swz64(d * 64 + 16g + 4t).
    const uint32_t dst0 = (uint32_t)(c >> 2) * (HD * BN) + (uint32_t)(16 * (c & 3)) * 64 + 4 * t4;
    const int total = max(rg.end - rg.start, 0);
    int b = rg.b, excl = rg.excl;
    int sb = (b < B) ? seq[b] : 0;
    int incl = excl + (sb + BN - 1) / BN;
    auto locate = [&](int gi) -> int {
        while (gi >= incl && b + 1 < B) { ++b; excl = incl; sb = seq[b]; incl = excl + (sb + BN - 1) / BN; }
        const int t = max(min((gi - excl) * BN + prow, sb - 1), 0);
        const int* ptr = pt + (long long)b * maxp;
        return PAGE1 ? ptr[t] : ptr[t / page] * page + t % page;
    };
    // Page 1: warp 0 stages each tile's page-table entries NV_PFD tiles ahead in sm.ptr (cp.async in the gathers'
    // groups), so a tile's gathers wait on shared memory, not on a DRAM round trip.
    auto stage = [&](int j) {
        const int gi = rg.start + j;
        while (gi >= incl && b + 1 < B) { ++b; excl = incl; sb = seq[b]; incl = excl + (sb + BN - 1) / BN; }
        const int kb = (gi - excl) * BN;
        if (tid < 16 && kb + 4 * tid < sb) cp16ca(smem_u32(&sm.ptr[j & 7][4 * tid]), pt + (long long)b * maxp + kb + 4 * tid);
        if (tid == 0) sm.lim[j & 7] = sb - 1 - kb;
    };
    int row = 0;
    if constexpr (PAGE1) {
        if (tid < 32)
            for (int j = 0; j < min(NV_PFD, total); ++j) stage(j);
        cp_commit();
        cp_wait<0>();
        asm volatile("bar.sync 1, 128;" ::: "memory");
    } else {
        row = total > 0 ? locate(rg.start) : 0;
    }
    // Page 1: 8 lanes per row (whole 128-byte lines per request), rows tid / 8 + 16j; else the row-pair mapping.
    const int grow = PAGE1 ? (tid >> 3) : prow;
    const uint32_t kdst = smem_u32(&sm.k[0][0]) + grow * 128, vdst = smem_u32(&sm.vs[0][0]) + grow * 128;
    for (int i = 0; i < total + NV_LAG; ++i) {
        if (i < total) {
            const int s = i % NS;
            if constexpr (PAGE1) {
                const int lim = sm.lim[i & 7];
                int cr[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) cr[j] = sm.ptr[i & 7][min(grow + 16 * j, lim)];
                if (tid < 32 && i + NV_PFD < total) stage(i + NV_PFD);
                if (NV_ABL != 5) mb_wait(&sm.empty[s], ((i / NS) & 1) ^ 1);
                const int ch = tid & 7;
                if (NV_ABL != 6 || i < NS)
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int r = grow + 16 * j;
                    const long long off = (long long)cr[j] * 128 + ch * 16;
                    cp16(kdst + s * (BN * 128) + 16 * j * 128 + ((ch ^ (r & 7)) << 4), kc + off);
                    cp16(vdst + s * (BN * 128) + 16 * j * 128 + ((ch ^ vsw(r)) << 4), vc + off);
                }
            } else {
                const int cur = row;
                if (i + 1 < total) row = locate(rg.start + i + 1);
                if (NV_ABL != 5) mb_wait(&sm.empty[s], ((i / NS) & 1) ^ 1);
                const uint8_t* ks = kc + (long long)cur * 128;
                const uint8_t* vs = vc + (long long)cur * 128;
                const uint32_t ksw = prow & 7, vsx = vsw(prow);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int ch = pc0 + j;
                    cp16(kdst + s * (BN * 128) + ((ch ^ ksw) << 4), ks + ch * 16);
                    cp16(vdst + s * (BN * 128) + ((ch ^ vsx) << 4), vs + ch * 16);
                }
            }
        }
        cp_commit();
        if (i >= NV_LAG) {
            const int s = (i - NV_LAG) % NS;
            cp_wait<NV_LAG>();
            asm volatile("bar.sync 1, 128;" ::: "memory");
            if (NV_ABL != 3 && (NV_ABL != 6 || i - NV_LAG < NS)) {
                // FP8 V -> FP16: thread converts 16-byte chunk c = tid & 7 (head c >> 2, d 16 (c & 3) + 0..15) of
                // rows tid / 8 + 16 j into FP16 chunks 2 (c & 3), +1 of the head's SW128 [key][d] tile.
                const uint32_t vsb = smem_u32(&sm.vs[s][0]);
                const int c = tid & 7, hh = c >> 2, cp = 2 * (c & 3);
                const uint32_t vhb = smem_u32(&sm.vh[s][hh][0]);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int r = (tid >> 3) + 16 * j;
                    uint4 x;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(x.x), "=r"(x.y), "=r"(x.z), "=r"(x.w)
                                 : "r"(vsb + r * 128 + ((c ^ vsw(r)) << 4)));
                    const uint4 y0 = make_uint4(cvt8(x.x), cvt8(x.x >> 16), cvt8(x.y), cvt8(x.y >> 16));
                    const uint4 y1 = make_uint4(cvt8(x.z), cvt8(x.z >> 16), cvt8(x.w), cvt8(x.w >> 16));
                    // (Head 1 stores its odd chunk first: the two heads' tiles alias banks.)
                    const uint4 ya = hh ? y1 : y0, yb = hh ? y0 : y1;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vhb + r * 128 + (((cp + hh) ^ (r & 7)) << 4)),
                                 "r"(ya.x), "r"(ya.y), "r"(ya.z), "r"(ya.w) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vhb + r * 128 + (((cp + 1 - hh) ^ (r & 7)) << 4)),
                                 "r"(yb.x), "r"(yb.y), "r"(yb.z), "r"(yb.w) : "memory");
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            if ((tid & 31) == 0) mb_arrive(&sm.ready[s]);
        }
    }
}

// Online softmax of one 64-key tile (rows r0 = regs 0,1 / r1 = regs 2,3 of each n8 block); P as FP16 A fragments
// (k16 step kk: regs kk*4 + 0/1 from n8 block 2kk, kk*4 + 2/3 from 2kk + 1). Returns true if the accumulators need
// alpha (the warp took the full update).
__device__ __forceinline__ bool softmax(const float (&s)[32], float (&m)[2], float (&l)[2], uint32_t (&p)[16],
                                        float (&alpha)[2], float scale) {
    if (NV_LAZY) {
        const float o0 = m[0], o1 = m[1];
        float l0 = 0.f, l1 = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float a0 = ex2(fmaf(s[j * 4 + 0], scale, -o0)), a1 = ex2(fmaf(s[j * 4 + 1], scale, -o0));
            const float b0 = ex2(fmaf(s[j * 4 + 2], scale, -o1)), b1 = ex2(fmaf(s[j * 4 + 3], scale, -o1));
            l0 += a0 + a1; l1 += b0 + b1;
            p[(j >> 1) * 4 + (j & 1) * 2 + 0] = pack_h2(a0, a1);
            p[(j >> 1) * 4 + (j & 1) * 2 + 1] = pack_h2(b0, b1);
        }
        // (Also false for inf/NaN sums, e.g. the first tile's m = -1e30.)
        const bool ok = (l0 <= NV_TH) && (l1 <= NV_TH);
        if (__all_sync(0xffffffff, ok)) { l[0] += l0; l[1] += l1; return false; }
    }
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
    const float mn0 = fmaxf(m[0], mx0 * scale), mn1 = fmaxf(m[1], mx1 * scale);
    alpha[0] = ex2(m[0] - mn0); alpha[1] = ex2(m[1] - mn1);
    m[0] = mn0; m[1] = mn1;
    float l0 = 0.f, l1 = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float a0 = ex2(fmaf(s[j * 4 + 0], scale, -mn0)), a1 = ex2(fmaf(s[j * 4 + 1], scale, -mn0));
        const float b0 = ex2(fmaf(s[j * 4 + 2], scale, -mn1)), b1 = ex2(fmaf(s[j * 4 + 3], scale, -mn1));
        l0 += a0 + a1; l1 += b0 + b1;
        p[(j >> 1) * 4 + (j & 1) * 2 + 0] = pack_h2(a0, a1);
        p[(j >> 1) * 4 + (j & 1) * 2 + 1] = pack_h2(b0, b1);
    }
    l[0] = l[0] * alpha[0] + l0; l[1] = l[1] * alpha[1] + l1;
    return true;
}
__device__ __forceinline__ void rescale(float (&o)[32], const float (&alpha)[2]) {
#pragma unroll
    for (int j = 0; j < 8; ++j) { o[j * 4 + 0] *= alpha[0]; o[j * 4 + 1] *= alpha[0]; o[j * 4 + 2] *= alpha[1]; o[j * 4 + 3] *= alpha[1]; }
}
__device__ __forceinline__ void issue_qk(float (&s)[32], uint32_t qa, uint32_t ka) {
    wg_fence();
    qk_k32(s, mdesc(qa, 512, 2), mdesc(ka, 1024, 1), 0);
    qk_k32(s, mdesc(qa + 32, 512, 2), mdesc(ka + 32, 1024, 1), 1);
    wg_commit();
}
__device__ __forceinline__ void issue_pv(float (&o)[32], const uint32_t (&p)[16], uint32_t va) {
    wg_fence();
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) pv_k16(o, p[kk * 4 + 0], p[kk * 4 + 1], p[kk * 4 + 2], p[kk * 4 + 3], mdesc(va + kk * 2048, 1024, 1));
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

template <int NS>
__device__ __forceinline__ void consumer(Smem<NS>& sm, int h, const __nv_bfloat16* __restrict__ qkv,
                                         const int* __restrict__ seq, Range rg, int B, int p, int nid, int qkv_w,
                                         float scale, float* __restrict__ pacc, float* __restrict__ pml) {
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
            uint8_t* qd = &sm.q[h][0];
            *reinterpret_cast<uint4*>(qd + swz64(r * 64 + half * 32)) = make_uint4(pk[0], pk[1], pk[2], pk[3]);
            *reinterpret_cast<uint4*>(qd + swz64(r * 64 + half * 32 + 16)) = make_uint4(pk[4], pk[5], pk[6], pk[7]);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("bar.sync %0, 128;" ::"r"(2 + h) : "memory");
        }
        float o[32], s[32];
#pragma unroll
        for (int i = 0; i < 32; ++i) o[i] = 0.f;
        float m[2] = {-1e30f, -1e30f}, l[2] = {0.f, 0.f}, alpha[2];
        uint32_t pa[16];
        if (NV_ABL >= 2 && NV_ABL <= 4) {
            for (int i = 0; i < n; ++i) { mb_wait(&sm.ready[(it + i) % NS], ((it + i) / NS) & 1); if (lane == 0) mb_arrive(&sm.empty[(it + i) % NS]); }
        } else {
            for (int i = 0; i < n; ++i) {
                const int sc = (it + i) % NS;
                if (NV_ABL != 1 && NV_ABL != 5) mb_wait(&sm.ready[sc], ((it + i) / NS) & 1);
                issue_qk(s, qa, smem_u32(&sm.k[sc][0]) + h * 64);
                wg_wait<0>();
                fence_regs(s);
                const int nv = hi - lo - i * BN;
                if (nv < BN) mask(s, lane, nv);
                if (softmax(s, m, l, pa, alpha, scale)) rescale(o, alpha);
                fence_regs(o);
                fence_regs(pa);
                issue_pv(o, pa, smem_u32(&sm.vh[sc][h][0]));
                wg_wait<0>();
                fence_regs(o);
                if (lane == 0) mb_arrive(&sm.empty[sc]);
            }
        }
        it += n;
        l[0] += __shfl_xor_sync(0xffffffff, l[0], 1); l[0] += __shfl_xor_sync(0xffffffff, l[0], 2);
        l[1] += __shfl_xor_sync(0xffffffff, l[1], 1); l[1] += __shfl_xor_sync(0xffffffff, l[1], 2);
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
            const float off = 0.f;
            *reinterpret_cast<float2*>(pml + (part + ra) * 2) = make_float2(m[0] - off, l[0]);
            *reinterpret_cast<float2*>(pml + (part + ra + 8) * 2) = make_float2(m[1] - off, l[1]);
        }
        gi = excl + t1;
        excl += nb;
        ++b;
        if (excl >= end) break;
    }
}

template <int NS, bool PAGE1>
__global__ void __launch_bounds__(384, 2) nverify(const __nv_bfloat16* __restrict__ qkv, const uint8_t* __restrict__ kc,
                                                  const uint8_t* __restrict__ vc, const float* __restrict__ ks,
                                                  const int* __restrict__ pt, const int* __restrict__ seq,
                                                  float* __restrict__ pacc, float* __restrict__ pml, float qk_scale,
                                                  int B, int P, int nid, int maxp, int page, int qkv_w) {
    extern __shared__ __align__(1024) uint8_t smraw[];
    Smem<NS>& sm = *reinterpret_cast<Smem<NS>*>((reinterpret_cast<uintptr_t>(smraw) + 1023) & ~uintptr_t(1023));
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    const int tid = threadIdx.x, wg = tid >> 7;
    if (tid == 0) {
        for (int i = 0; i < NS; ++i) { mb_init(&sm.ready[i], 4); mb_init(&sm.empty[i], 8); }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    const int p = blockIdx.x;
    const Range rg = bal(seq, B, p, P);
    if (wg == 0) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(NV_PREGS));
        if (NV_ABL != 1) producer<NS, PAGE1>(sm, kc, vc, pt, seq, rg, B, maxp, page);
    } else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(NV_CREGS));
        consumer<NS>(sm, wg - 1, qkv, seq, rg, B, p, nid, qkv_w, qk_scale * ks[0], pacc, pml);
    }
    // Launched alongside _new_tokens: finish only after it, so the merge (waiting on this grid) sees its partials.
    asm volatile("griddepcontrol.wait;" ::: "memory");
}
}  // namespace nv

template <int NS, bool PAGE1>
static void nv_launch(torch::Tensor qkv, torch::Tensor kc, torch::Tensor vc, torch::Tensor ks, torch::Tensor pt,
                      torch::Tensor seq, torch::Tensor pacc, torch::Tensor pml, double qk_scale, int64_t P,
                      int64_t nid, int64_t page) {
    auto f = nv::nverify<NS, PAGE1>;
    const int smem = (int)sizeof(nv::Smem<NS>) + 1024;
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
void nverify(torch::Tensor qkv, torch::Tensor kc, torch::Tensor vc, torch::Tensor ks, torch::Tensor pt,
             torch::Tensor seq, torch::Tensor pacc, torch::Tensor pml, double qk_scale, int64_t P, int64_t nid,
             int64_t page) {
    if (page == 1) nv_launch<NV_NS, true>(qkv, kc, vc, ks, pt, seq, pacc, pml, qk_scale, P, nid, page);
    else nv_launch<NV_NS, false>(qkv, kc, vc, ks, pt, seq, pacc, pml, qk_scale, P, nid, page);
}

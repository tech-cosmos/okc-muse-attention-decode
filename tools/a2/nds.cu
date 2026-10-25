// agent2 nds (decode sliding, both KV heads in one consumer warpgroup) built on nvs: native verify-SLIDING attention (L 8, G 8, D 64, HKV 2, page 1), from tools/s71/nverify.cu.
// One CTA per (split, request): warpgroup 0 gathers both KV heads' K/V rows (128-B lines) with cp.async and
// transposes V to FP8 V^T; warpgroups 1/2 (KV head 0/1) normalize + RoPE their queries, run QK^T (FP8 wgmma),
// the online softmax and P@V with P in E4M3 (FA3's FP8 path), QK(i) overlapping PV(i-1). Partials in
// _attn_split's layout [B, HKV, S, 64, 64] FP32 for _combine_hb. NVS_NEW: split S-1 also attends the 8 new tokens.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <stdint.h>

#ifndef NV_NS
#define NV_NS 4
#endif
#ifndef NV_PREGS
#define NV_PREGS 40
#endif
#ifndef NV_CREGS
#define NV_CREGS 208
#endif
#ifndef NV_MINB
#define NV_MINB 1
#endif
namespace nv {
constexpr int BN = 64, HD = 64, MR = 64;
#ifndef NV_LAG
#define NV_LAG 1
#endif
#ifndef NV_PFD
#define NV_PFD 4
#endif
#ifndef NV_TR
#define NV_TR 1
#endif
#ifndef NV_LAZY
#define NV_LAZY 0
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
    alignas(1024) uint8_t vt[NS][2][HD * BN];  // per head V^T [d][key slot], SW64 (wgmma B, K-major), keys permuted
    alignas(1024) uint8_t q[2][MR * HD];       // per head FP8 queries [row][d], SW64
    int ptr[8][BN];                            // page 1: page-table entries of tiles i .. i + NV_PFD (ring of 8)
    int lim[8];                                // and each tile's last valid key
    uint64_t ready[NS], empty[NS];
    alignas(1024) uint8_t knew[BN * 128];      // last split: the 8 new keys as a K tile (rows 0-7; rows 8-63 zero)
    alignas(1024) uint8_t vtnew[2][HD * BN];   // and their V^T (key slots 0-7 of group 0; the rest zero)
};
// Staging swizzle: SW128 (r & 7) for the ldmatrix transpose (8 consecutive rows per phase), else conflict-free for
// the 4-row register transpose.
__device__ __forceinline__ uint32_t vsw(int r) { return NV_TR ? (r & 7) : (((r >> 1) & 3) | (((r >> 4) & 1) << 2)); }


struct Range { int b, lo, hi, n, seq; };
__device__ __forceinline__ Range rng(const int* __restrict__ seq, int b, int split, int ch, int win) {
    Range r;
    r.b = b;
    r.seq = seq[b];
    const int base = win > 0 ? max(r.seq - (win - 1), 0) : 0;
    r.lo = base + split * ch;
    r.hi = min(r.lo + ch, r.seq);
    r.n = r.hi > r.lo ? (r.hi - r.lo + BN - 1) / BN : 0;
    return r;
}
__device__ __forceinline__ void cp4ca(uint32_t dst, const void* src) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(dst), "l"(src) : "memory");
}

template <int NS>
__device__ __forceinline__ void producer(Smem<NS>& sm, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
                                         const int* __restrict__ pt, Range rg, int maxp) {
    const int tid = threadIdx.x;  // 0..127
    const int total = rg.n;
    const int* ptr = pt + (long long)rg.b * maxp;
    // Warp 0 stages each tile's page-table entries NV_PFD tiles ahead in sm.ptr (2 x 4-byte cp.async per lane: the
    // window start is not 16-byte aligned).
    auto stage = [&](int j) {
        const int kb = rg.lo + j * BN;
        if (kb + tid < rg.hi) cp4ca(smem_u32(&sm.ptr[j & 7][tid]), ptr + kb + tid);
        if (kb + tid + 32 < rg.hi) cp4ca(smem_u32(&sm.ptr[j & 7][tid + 32]), ptr + kb + tid + 32);
        if (tid == 0) sm.lim[j & 7] = rg.hi - 1 - kb;
    };
    if (tid < 32)
        for (int j = 0; j < min(NV_PFD, total); ++j) stage(j);
    cp_commit();
    cp_wait<0>();
    asm volatile("bar.sync 1, 128;" ::: "memory");
    const int grow = tid >> 3;
    const uint32_t kdst = smem_u32(&sm.k[0][0]) + grow * 128, vdst = smem_u32(&sm.vs[0][0]) + grow * 128;
    const int lane = tid & 31;
    for (int i = 0; i < total + NV_LAG; ++i) {
        if (i < total) {
            const int s = i % NS;
            const int lim = sm.lim[i & 7];
            int cr[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) cr[j] = sm.ptr[i & 7][min(grow + 16 * j, lim)];
            if (tid < 32 && i + NV_PFD < total) stage(i + NV_PFD);
            mb_wait(&sm.empty[s], ((i / NS) & 1) ^ 1);
            const int ch = tid & 7;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int r = grow + 16 * j;
                const long long off = (long long)cr[j] * 128 + ch * 16;
                cp16(kdst + s * (BN * 128) + 16 * j * 128 + ((ch ^ (r & 7)) << 4), kc + off);
                cp16(vdst + s * (BN * 128) + 16 * j * 128 + ((ch ^ vsw(r)) << 4), vc + off);
            }
        }
        cp_commit();
        if (i >= NV_LAG) {
            const int s = (i - NV_LAG) % NS;
            cp_wait<NV_LAG>();
            asm volatile("bar.sync 1, 128;" ::: "memory");
            {
                const uint32_t vsb = smem_u32(&sm.vs[s][0]), vtb = smem_u32(&sm.vt[s][0][0]);
                const int wq = tid >> 5, mi = lane >> 3, ri = lane & 7;
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int u = wq * 4 + q, gg = u & 3, hc = u >> 2;
                    const int r = 16 * gg + (mi & 1) * 8 + ri, cc = 2 * hc + (mi >> 1);
                    uint32_t R0, R1, R2, R3;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                                 : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "r"(vsb + r * 128 + ((cc ^ (r & 7)) << 4)));
                    const uint32_t S0 = prmt(R0, R1, 0x6420), S1 = prmt(R0, R1, 0x7531);
                    const uint32_t S2 = prmt(R2, R3, 0x6420), S3 = prmt(R2, R3, 0x7531);
                    const int d = 32 * (hc & 1) + 16 * (mi >> 1) + 2 * ri + (mi & 1);
                    const uint32_t o = (uint32_t)(hc >> 1) * (HD * BN) + swz64(d * 64 + 16 * gg);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1,%2,%3,%4};"
                                 :: "r"(vtb + o), "r"(S0), "r"(S1), "r"(S2), "r"(S3) : "memory");
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            if ((tid & 31) == 0) mb_arrive(&sm.ready[s]);
        }
    }
}

__device__ __forceinline__ void softmax(float (&s)[32], float (&m)[2], float (&l)[2], uint32_t (&p)[8],
                                        float (&alpha)[2], float scale) {
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
    if (NV_LAZY) {
        bool grow = (mn0 > m[0] + 8.f) || (mn1 > m[1] + 8.f);
        if (!__any_sync(0xffffffff, grow)) { mn0 = m[0]; mn1 = m[1]; }
    }
    alpha[0] = ex2(m[0] - mn0); alpha[1] = ex2(m[1] - mn1);
    m[0] = mn0; m[1] = mn1;
    const float off = NV_LAZY ? 0.f : 8.f;
    const float o0 = mn0 - off, o1 = mn1 - off;
    float e[32], l0 = 0.f, l1 = 0.f;
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
}
__device__ __forceinline__ void rescale(float (&o)[32], const float (&alpha)[2]) {
    if (NV_LAZY && __all_sync(0xffffffff, alpha[0] == 1.f && alpha[1] == 1.f)) return;
#pragma unroll
    for (int j = 0; j < 8; ++j) { o[j * 4 + 0] *= alpha[0]; o[j * 4 + 1] *= alpha[0]; o[j * 4 + 2] *= alpha[1]; o[j * 4 + 3] *= alpha[1]; }
}
__device__ __forceinline__ void issue_qk(float (&s)[32], uint32_t qa, uint32_t ka) {
    wg_fence();
    qk_k32(s, mdesc(qa, 512, 2), mdesc(ka, 1024, 1), 0);
    qk_k32(s, mdesc(qa + 32, 512, 2), mdesc(ka + 32, 1024, 1), 1);
    wg_commit();
}
__device__ __forceinline__ void issue_pv(float (&o)[32], const uint32_t (&p)[8], uint32_t va) {
    wg_fence();
    pv8_k32(o, p[0], p[1], p[2], p[3], mdesc(va, 512, 2));
    pv8_k32(o, p[4], p[5], p[6], p[7], mdesc(va + 32, 512, 2));
    wg_commit();
}
// Score masks: key column col (of this tile) is out if col >= nvalid (past the split's end) or col < thr0 / thr1
// (rows of token 2w / 2w + 1: before the sliding window's start).
__device__ __forceinline__ void mask(float (&s)[32], int lane, int nvalid, int thr0) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int col = j * 8 + (lane & 3) * 2;
        if (col >= nvalid || col < thr0) s[j * 4 + 0] = -INFINITY;
        if (col + 1 >= nvalid || col + 1 < thr0) s[j * 4 + 1] = -INFINITY;
        if (col >= nvalid || col < thr0 + 1) s[j * 4 + 2] = -INFINITY;
        if (col + 1 >= nvalid || col + 1 < thr0 + 1) s[j * 4 + 3] = -INFINITY;
    }
}


#define D64(a) D32(a), "+f"(a[32]), "+f"(a[33]), "+f"(a[34]), "+f"(a[35]), "+f"(a[36]), "+f"(a[37]), "+f"(a[38]), \
    "+f"(a[39]), "+f"(a[40]), "+f"(a[41]), "+f"(a[42]), "+f"(a[43]), "+f"(a[44]), "+f"(a[45]), "+f"(a[46]), "+f"(a[47]), \
    "+f"(a[48]), "+f"(a[49]), "+f"(a[50]), "+f"(a[51]), "+f"(a[52]), "+f"(a[53]), "+f"(a[54]), "+f"(a[55]), "+f"(a[56]), \
    "+f"(a[57]), "+f"(a[58]), "+f"(a[59]), "+f"(a[60]), "+f"(a[61]), "+f"(a[62]), "+f"(a[63])
#define R64 "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27," \
    "%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55," \
    "%56,%57,%58,%59,%60,%61,%62,%63}"
__device__ __forceinline__ void pv16_k32(float (&o)[64], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3 " R64 ", {%64,%65,%66,%67}, %68, 1, 1, 1;\n"
                 : D64(o) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
}
// QK over the whole 128-B K row (both heads; each query row is zero in the other head's 64 d).
__device__ __forceinline__ void issue_qk2(float (&s)[32], uint32_t qa, uint32_t ka) {
    wg_fence();
#pragma unroll
    for (int k = 0; k < 4; ++k) qk_k32(s, mdesc(qa + 32 * k, 1024, 1), mdesc(ka + 32 * k, 1024, 1), k);
    wg_commit();
}
// P@V into both heads' 128 output columns (V^T of head 0 then head 1: 128 SW64 rows).
__device__ __forceinline__ void issue_pv2(float (&o)[64], const uint32_t (&p)[8], uint32_t va) {
    wg_fence();
    pv16_k32(o, p[0], p[1], p[2], p[3], mdesc(va, 512, 2));
    pv16_k32(o, p[4], p[5], p[6], p[7], mdesc(va + 32, 512, 2));
    wg_commit();
}
__device__ __forceinline__ void dmask(float (&s)[32], int lane, int nvalid, int thr) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int col = j * 8 + (lane & 3) * 2;
        if (col >= nvalid || col < thr) { s[j * 4 + 0] = -INFINITY; s[j * 4 + 2] = -INFINITY; }
        if (col + 1 >= nvalid || col + 1 < thr) { s[j * 4 + 1] = -INFINITY; s[j * 4 + 3] = -INFINITY; }
    }
}
// Row lane / 4 is head 0 (output columns 0-63), row lane / 4 + 8 head 1 (columns 64-127).
__device__ __forceinline__ void drescale(float (&o)[64], const float (&alpha)[2]) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        o[j * 4 + 0] *= alpha[0]; o[j * 4 + 1] *= alpha[0];
        o[32 + j * 4 + 2] *= alpha[1]; o[32 + j * 4 + 3] *= alpha[1];
    }
}

// Decode (L 1, G 8): one consumer warpgroup for both KV heads. m64 rows 0-7 = head 0's 8 query heads (d 0-63 of the
// K row; 64-127 zero), rows 8-15 = head 1's (d 64-127), rows 16-63 zero: only warp 0 has live rows and runs the
// softmax; warps 1-3 just take part in the wgmmas (P = 0).
template <int NS>
__device__ __forceinline__ void dconsumer(Smem<NS>& sm, const __nv_bfloat16* __restrict__ qkv,
                                          const float* __restrict__ cs, Range rg, int split, int S, int qkv_w,
                                          float scale, int win, float* __restrict__ pacc, float* __restrict__ pml,
                                          uint8_t* __restrict__ kc, uint8_t* __restrict__ vc,
                                          const long long* __restrict__ loc, float ksv, float vsv, __nv_bfloat16* __restrict__ out) {
    const int tid = threadIdx.x & 127, lane = tid & 31, w = tid >> 5;
    uint8_t* qd = &sm.q[0][0];  // [64 rows][128 B] SW128
    const uint32_t qa = smem_u32(qd);
    const int n = rg.n, lo = rg.lo, hi = rg.hi, b = rg.b;
    const bool last_split = split == S - 1;
    {
        const uint4 z = make_uint4(0, 0, 0, 0);
        if (last_split) {
            uint4* vz = reinterpret_cast<uint4*>(&sm.vtnew[0][0]);
#pragma unroll
            for (int j = 0; j < 4; ++j) vz[tid + 128 * j] = z;
        }
        if (w > 0) {
            uint4* qz = reinterpret_cast<uint4*>(qd + 16 * 128);
#pragma unroll
            for (int j = 0; j < 4; ++j) qz[tid - 32 + 96 * j] = z;
        }
        asm volatile("bar.sync 2, 128;" ::: "memory");
    }
    if (w == 0) {   // Queries: row r = query head r (KV head r / 8); 2 lanes per row (32 d each).
        const int r = lane >> 1, half = lane & 1, h = r >> 3;
        const int pos = rg.seq;
        const __nv_bfloat16* src = qkv + (long long)b * qkv_w + r * HD + half * 32;
        uint4 raw[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) raw[j] = reinterpret_cast<const uint4*>(src)[j];
        float4 cv[4], sv[4];
        const float4* cp = reinterpret_cast<const float4*>(cs + (long long)pos * HD + half * 16);
#pragma unroll
        for (int j = 0; j < 4; ++j) { cv[j] = cp[j]; sv[j] = cp[j + 8]; }
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
        const float* cf = reinterpret_cast<const float*>(cv);
        const float* sf = reinterpret_cast<const float*>(sv);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float a = x[2 * i], bb = x[2 * i + 1];
            x[2 * i] = (a * cf[i] - bb * sf[i]) * rs;
            x[2 * i + 1] = (bb * cf[i] + a * sf[i]) * rs;
        }
        uint32_t pk[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) pk[j] = pack_e4(x[4 * j], x[4 * j + 1], x[4 * j + 2], x[4 * j + 3]);
        const int c0 = h * 4 + half * 2, z0 = (1 - h) * 4 + half * 2;  // 16-B chunks: live d, zero d
        *reinterpret_cast<uint4*>(qd + r * 128 + ((c0 ^ (r & 7)) << 4)) = make_uint4(pk[0], pk[1], pk[2], pk[3]);
        *reinterpret_cast<uint4*>(qd + r * 128 + (((c0 + 1) ^ (r & 7)) << 4)) = make_uint4(pk[4], pk[5], pk[6], pk[7]);
        *reinterpret_cast<uint4*>(qd + r * 128 + ((z0 ^ (r & 7)) << 4)) = make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4*>(qd + r * 128 + (((z0 + 1) ^ (r & 7)) << 4)) = make_uint4(0, 0, 0, 0);
        if (last_split) {
            // The step's new token: lane = KV head (lane / 16), d 4 (lane % 16) + 0..3: RMS-norm + RoPE, E4M3 (as
            // _new), stored into the cache and into the new-token K row 0 / V^T key slot 0.
            const int hh = lane >> 4, d0 = (lane & 15) * 4, npos = rg.seq;
            const __nv_bfloat16* kp = qkv + (long long)b * qkv_w + (16 + hh) * HD + d0;
            const uint2 kr = *reinterpret_cast<const uint2*>(kp), vr = *reinterpret_cast<const uint2*>(kp + 2 * HD);
            const float2 c2 = *reinterpret_cast<const float2*>(cs + (long long)npos * HD + d0 / 2);
            const float2 s2 = *reinterpret_cast<const float2*>(cs + (long long)npos * HD + 32 + d0 / 2);
            const long long slot = loc[b];
            const float2 k01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&kr.x));
            const float2 k23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&kr.y));
            const float2 v01 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vr.x));
            const float2 v23 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&vr.y));
            float kss = k01.x * k01.x + k01.y * k01.y + k23.x * k23.x + k23.y * k23.y;
#pragma unroll
            for (int o = 1; o < 16; o <<= 1) kss += __shfl_xor_sync(0xffffffff, kss, o);
            const float krs = rsqrtf(kss / HD + 1e-5f);
            const uint32_t kq = pack_e4((k01.x * c2.x - k01.y * s2.x) * krs / ksv, (k01.y * c2.x + k01.x * s2.x) * krs / ksv,
                                        (k23.x * c2.y - k23.y * s2.y) * krs / ksv, (k23.y * c2.y + k23.x * s2.y) * krs / ksv);
            const uint32_t vq = pack_e4(v01.x / vsv, v01.y / vsv, v23.x / vsv, v23.y / vsv);
            *reinterpret_cast<uint32_t*>(kc + slot * 128 + hh * HD + d0) = kq;
            *reinterpret_cast<uint32_t*>(vc + slot * 128 + hh * HD + d0) = vq;
            const int c = hh * 4 + (d0 >> 4);
            *reinterpret_cast<uint32_t*>(&sm.knew[(c << 4) + (d0 & 15)]) = kq;
#pragma unroll
            for (int e = 0; e < 4; ++e) sm.vtnew[hh][swz64((d0 + e) * 64)] = (uint8_t)(vq >> (8 * e));
        }
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("bar.sync 2, 128;" ::: "memory");
    float o[64], s[32];
#pragma unroll
    for (int i = 0; i < 64; ++i) o[i] = 0.f;
    float m[2] = {-1e30f, -1e30f}, l[2] = {0.f, 0.f}, alpha[2];
    uint32_t pa[8], pb[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { pa[i] = 0u; pb[i] = 0u; }
    const bool live = w == 0;
    // Window start relative to key lo: keys t < seq - (win - 1) are out.
    const int thr = win > 0 ? rg.seq - (win - 1) - lo : -(1 << 30);
    bool newdone = false;
    if (last_split) {
        issue_qk2(s, qa, smem_u32(&sm.knew[0]));
        wg_wait<0>();
        fence_regs(s);
        if (live) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int col = j * 8 + (lane & 3) * 2;
                if (col > 0) { s[j * 4 + 0] = -INFINITY; s[j * 4 + 2] = -INFINITY; }
                s[j * 4 + 1] = -INFINITY; s[j * 4 + 3] = -INFINITY;
            }
            softmax(s, m, l, pa, alpha, scale);
        }
        fence_regs(pa);
        issue_pv2(o, pa, smem_u32(&sm.vtnew[0][0]));
        wg_wait<0>();
        fence_regs(o);
        newdone = true;
    }
    if (n > 0) {
        mb_wait(&sm.ready[0], 0);
        issue_qk2(s, qa, smem_u32(&sm.k[0][0]));
        wg_wait<0>();
        fence_regs(s);
        if (live) {
            if (hi - lo < BN || thr > 0) dmask(s, lane, hi - lo, thr);
            softmax(s, m, l, pa, alpha, scale);
            if (newdone) { drescale(o, alpha); fence_regs(o); }
        }
        auto step = [&](uint32_t (&pin)[8], uint32_t (&pout)[8], int i) {
            const int sp = (i - 1) % NS, sc = i % NS;
            mb_wait(&sm.ready[sc], (i / NS) & 1);
            issue_qk2(s, qa, smem_u32(&sm.k[sc][0]));
            fence_regs(pin);
            issue_pv2(o, pin, smem_u32(&sm.vt[sp][0][0]));
            wg_wait<1>();
            fence_regs(s);
            const int nv = hi - lo - i * BN;
            if (live) {
                if (nv < BN || thr - i * BN > 0) dmask(s, lane, nv, thr - i * BN);
                softmax(s, m, l, pout, alpha, scale);
            }
            wg_wait<0>();
            fence_regs(o);
            if (lane == 0) mb_arrive(&sm.empty[sp]);
            if (live) drescale(o, alpha);
            fence_regs(o);
        };
        auto lastt = [&](uint32_t (&pin)[8]) {
            const int sp = (n - 1) % NS;
            fence_regs(pin);
            issue_pv2(o, pin, smem_u32(&sm.vt[sp][0][0]));
            wg_wait<0>();
            fence_regs(o);
            if (lane == 0) mb_arrive(&sm.empty[sp]);
        };
        int i = 1;
        for (; i + 1 < n; i += 2) { step(pa, pb, i); step(pb, pa, i + 1); }
        if (i < n) { step(pa, pb, i); lastt(pb); } else lastt(pa);
    }
    if (!live) return;
    l[0] += __shfl_xor_sync(0xffffffff, l[0], 1); l[0] += __shfl_xor_sync(0xffffffff, l[0], 2);
    l[1] += __shfl_xor_sync(0xffffffff, l[1], 1); l[1] += __shfl_xor_sync(0xffffffff, l[1], 2);
    const int ra = lane >> 2;
    const long long part0 = (((long long)b * 2 + 0) * S + split) * 8, part1 = (((long long)b * 2 + 1) * S + split) * 8;
    float* pa0 = pacc + (part0 + ra) * HD;
    float* pa1 = pacc + (part1 + ra) * HD;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int col = j * 8 + (lane & 3) * 2;
        *reinterpret_cast<float2*>(pa0 + col) = make_float2(o[j * 4 + 0], o[j * 4 + 1]);
        *reinterpret_cast<float2*>(pa1 + col) = make_float2(o[32 + j * 4 + 2], o[32 + j * 4 + 3]);
    }
    if ((lane & 3) == 0) {
        const float off = NV_LAZY ? 0.f : 8.f;
        *reinterpret_cast<float2*>(pml + (part0 + ra) * 2) = make_float2(m[0] - off, l[0]);
        *reinterpret_cast<float2*>(pml + (part1 + ra) * 2) = make_float2(m[1] - off, l[1]);
    }
}

template <int NS>
#ifdef NV_MAXR
__global__ void __maxnreg__(NV_MAXR) nds(
#else
__global__ void __launch_bounds__(256, NV_MINB) nds(
#endif
const __nv_bfloat16* __restrict__ qkv, const float* __restrict__ cs,
                                              uint8_t* __restrict__ kc, uint8_t* __restrict__ vc,
                                              const float* __restrict__ ks, const float* __restrict__ vsc,
                                              const long long* __restrict__ loc, const int* __restrict__ pt,
                                              const int* __restrict__ seq, float* __restrict__ pacc,
                                              float* __restrict__ pml, float qk_scale, int S, int ch, int maxp,
                                              int qkv_w, int win, __nv_bfloat16* __restrict__ out) {
    extern __shared__ __align__(1024) uint8_t smraw[];
    Smem<NS>& sm = *reinterpret_cast<Smem<NS>*>((reinterpret_cast<uintptr_t>(smraw) + 1023) & ~uintptr_t(1023));
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    const int tid = threadIdx.x, wg = tid >> 7;
    if (tid == 0) {
        for (int i = 0; i < NS; ++i) { mb_init(&sm.ready[i], 4); mb_init(&sm.empty[i], 4); }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    const int split = blockIdx.x % S;
    const Range rg = rng(seq, blockIdx.x / S, split, ch, win);
    if (wg == 0) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(NV_PREGS));
        producer<NS>(sm, kc, vc, pt, rg, maxp);
    } else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(NV_CREGS));
        dconsumer<NS>(sm, qkv, cs, rg, split, S, qkv_w, qk_scale * ks[0], win, pacc, pml, kc, vc, loc, ks[0], vsc[0], out);
    }
}
}  // namespace nv

void nds(torch::Tensor qkv, torch::Tensor cs, torch::Tensor kc, torch::Tensor vc, torch::Tensor ks, torch::Tensor vsc,
         torch::Tensor loc, torch::Tensor pt,
         torch::Tensor seq, torch::Tensor pacc, torch::Tensor pml, torch::Tensor out, double qk_scale, int64_t S, int64_t ch,
         int64_t win) {
    auto f = nv::nds<NV_NS>;
    const int smem = (int)sizeof(nv::Smem<NV_NS>) + 1024;
    static bool once = false;
    if (!once) { cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); once = true; }
    const int B = (int)seq.size(0);
    f<<<dim3((unsigned)(S * B)), 256, smem, c10::cuda::getCurrentCUDAStream()>>>(
        (const __nv_bfloat16*)qkv.data_ptr(), (const float*)cs.data_ptr(), (uint8_t*)kc.data_ptr(),
        (uint8_t*)vc.data_ptr(), (const float*)ks.data_ptr(), (const float*)vsc.data_ptr(),
        (const long long*)loc.data_ptr(), (const int*)pt.data_ptr(),
        (const int*)seq.data_ptr(), (float*)pacc.data_ptr(), (float*)pml.data_ptr(), (float)qk_scale, (int)S, (int)ch,
        (int)pt.size(1), (int)qkv.size(-1), (int)win, (__nv_bfloat16*)out.data_ptr());
}

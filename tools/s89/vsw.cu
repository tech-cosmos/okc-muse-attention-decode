// s89: verify (q_len 8) sliding attention as ONE kernel with FP8 wgmma.
// Per (request, KV head) one 16-CTA cluster; CTA rank p, warpgroup wg < 2 covers window keys
// w in [128 p + 64 wg, +64) (w = t - (seq - 2047); cached keys w < 2047) as one 64-key tile, single pass:
// S = Q K^T (m64n64k32 E4M3, Q/K in smem SW64), row max, P = exp2(S - (m - 8)) in E4M3 (FA3's FP8 path),
// O = P V (RS wgmma, V^T built in smem from register gathers, K-major SW64, key slots permuted to P's layout).
// Warpgroup 2 of CTA 15 runs the block's 8 new tokens (norm, RoPE, quantize, cache store, causal 8-key tile).
// Every warpgroup pushes its FP32 partials (o, m, l) straight to the owner CTAs (4 rows each; st.async +
// mbarrier complete_tx), which merge the 33 partials, RMS-normalize, gate and store BF16.
// Fixed shapes: HQ 16, HKV 2, D 64, window 2048, L 8.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace vsw {
constexpr int HQ = 16, HKV = 2, D = 64, W = 2048, L = 8, QKV_W = (2 * HQ + 2 * HKV) * D, NP = 16, RO = 4;
#ifndef VSW_BULK
#define VSW_BULK 1
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
__device__ __forceinline__ void qk8_k32(float (&s)[4], uint64_t da, uint64_t db, int acc) {
    asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %6, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0,%1,%2,%3}, %4, %5, p, 1, 1;\n}\n"
                 : "+f"(s[0]), "+f"(s[1]), "+f"(s[2]), "+f"(s[3]) : "l"(da), "l"(db), "r"(acc));
}
__device__ __forceinline__ void pv8_k32(float (&o)[32], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t db) {
    asm volatile("wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 " R32 ", {%32,%33,%34,%35}, %36, 1, 1, 1;\n"
                 : D32(o) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
}
__device__ __forceinline__ unsigned short pk2(float lo, float hi) {
    unsigned short r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
    return r;
}
__device__ __forceinline__ uint32_t pack_e4(float a, float b, float c, float d) {
    return (uint32_t)pk2(a, b) | ((uint32_t)pk2(c, d) << 16);
}
__device__ __forceinline__ float ex2(float x) { float y; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x)); return y; }
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t s) {
    uint32_t d; asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(s)); return d;
}
__device__ __forceinline__ unsigned mapa(unsigned addr, unsigned cta) {
    unsigned r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(cta));
    return r;
}
__device__ __forceinline__ void sa2(unsigned addr, unsigned mbar, float a, float b) {
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1,%2}, [%3];"
                 ::"r"(addr), "f"(a), "f"(b), "r"(mbar) : "memory");
}
__device__ __forceinline__ float bf(unsigned short x) { return __uint_as_float((unsigned)x << 16); }
__device__ __forceinline__ long slot_of(const int* ptr, int t, int page) {
    return page == 1 ? (long)ptr[t] : (long)ptr[t / page] * page + t % page;
}

__device__ __forceinline__ unsigned long long gtime() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ unsigned long long g_dbg[32 * 3 * 8];
#define STAMP(k) if (abl == 9 && t == 0) g_dbg[(blockIdx.x * 3 + wg) * 8 + (k)] = gtime();
__device__ float g_scr[8 * 2 * 16 * 16 * (4 * 64 + 8)];
struct Smem {
    alignas(1024) uint8_t q[3][64 * 64];   // per warpgroup: FP8 queries [row][d], SW64
    alignas(1024) uint8_t k[3][64 * 64];   // per warpgroup: K tile [key][d], SW64 (wg 2: the 8 new keys)
    alignas(1024) uint8_t vt[3][64 * 64];  // per warpgroup: V^T [d][key slot], SW64, slots permuted as P
    float racc[NP][RO * D + 2 * RO];       // received partials [source]: [owned row][d], then (m, l) per row
    float stage[16][RO * D + 2 * RO];      // the CTA's merged partial grouped by owner, for the bulk copies
    float xf[2][36][128];                  // warpgroups 1, 2 -> 0: o[32], mA, lA, mB, lB per thread
    unsigned long long mbar;
};

template <int PAGE>
__global__ void __launch_bounds__(384, 1) verify(const unsigned short* __restrict__ qkv, const float* __restrict__ cs,
                                                 unsigned char* kc, unsigned char* vc, const int* __restrict__ pt,
                                                 const int* __restrict__ seqs, const long* __restrict__ loc,
                                                 const float* __restrict__ ks, const float* __restrict__ vs,
                                                 __nv_bfloat16* __restrict__ out, int MP, float qk_scale, int spec,
                                                 int abl) {
    extern __shared__ __align__(1024) unsigned char smraw[];
    Smem& sm = *reinterpret_cast<Smem*>((reinterpret_cast<uintptr_t>(smraw) + 1023) & ~uintptr_t(1023));
    const int tid = threadIdx.x, wg = tid >> 7, t = tid & 127, w = t >> 5, lane = tid & 31, g = lane >> 2,
              c = lane & 3;
    unsigned rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    const int cid = blockIdx.x >> 4, b = cid >> 1, h = cid & 1;
    const unsigned mb = smem_u32(&sm.mbar);
    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n\tfence.mbarrier_init.release.cluster;" ::"r"(mb) : "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(mb), "r"(NP * (RO * D * 4 + RO * 8))
                     : "memory");
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    const bool newer = wg == 2;
    STAMP(0)
    if (newer && rank != 15) {
        asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
        return;
    }

    // Page-table-independent loads first. Q: thread row r = t >> 1 (token r >> 3, head h 8 + (r & 7)), d 32 half.
    const int qr = t >> 1, half = t & 1;
    const unsigned short* qp = qkv + (long)(b * L + (qr >> 3)) * QKV_W + (h * 8 + (qr & 7)) * D + 32 * half;
    uint4 qraw[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) qraw[j] = *(const uint4*)(qp + 8 * j);
    const float kscale = *ks, vscale = *vs;
    // New tokens (wg 2): thread token nt = t >> 4, d 4 (t & 15) .. + 3.
    const int nt = t >> 4, nd = 4 * (t & 15);
    uint2 nkw = make_uint2(0, 0), nvw = make_uint2(0, 0);
    long nloc = 0;
    if (newer) {
        const unsigned short* kp = qkv + (long)(b * L + nt) * QKV_W + (HQ + h) * D + nd;
        nkw = *(const uint2*)kp;
        nvw = *(const uint2*)(kp + HKV * D);
        nloc = loc[b * L + nt];
    }
    int seq = spec >= 0 ? spec : seqs[b];
    const int* ptr = pt + (long)b * MP;
    float4 cq[4], sq[4];
    float ncs[4];
    int base, w0, tmax;
    uint4 kr[2];
    uint2 vr[4];
    // K units u = t + 128 i: key u >> 2, 16-byte chunk u & 3. V unit: keys 16 (t >> 5) + 2 ((t >> 3) & 3) +
    // {0, 1, 8, 9}, 8 bytes d 8 (t & 7) .. + 7.
    const int vk0 = 16 * (t >> 5) + 2 * ((t >> 3) & 3), vdd = t & 7;
#define VSW_LOADS                                                                                              \
    {                                                                                                          \
        const float* cr = cs + (long)(seq + (qr >> 3)) * D + 16 * half;                                        \
        _Pragma("unroll") for (int j = 0; j < 4; ++j) { cq[j] = *(const float4*)(cr + 4 * j);                  \
                                                        sq[j] = *(const float4*)(cr + 32 + 4 * j); }           \
        if (newer) {                                                                                           \
            const float* cn = cs + (long)(seq + nt) * D + nd / 2;                                              \
            ncs[0] = cn[0]; ncs[1] = cn[1]; ncs[2] = cn[32]; ncs[3] = cn[33];                                   \
        }                                                                                                      \
        base = seq - (W - 1);                                                                                  \
        w0 = 128 * rank + 64 * wg;                                                                             \
        tmax = max(seq - 1, 0);                                                                                \
        if (!newer) {                                                                                          \
            long ksl[2], vsl[4];                                                                               \
            _Pragma("unroll") for (int i = 0; i < 2; ++i)                                                      \
                ksl[i] = slot_of(ptr, min(max(base + w0 + ((t + 128 * i) >> 2), 0), tmax), PAGE);              \
            _Pragma("unroll") for (int i = 0; i < 4; ++i)                                                      \
                vsl[i] = slot_of(ptr, min(max(base + w0 + vk0 + (i & 1) + 8 * (i >> 1), 0), tmax), PAGE);      \
            _Pragma("unroll") for (int i = 0; i < 2; ++i)                                                      \
                kr[i] = *(const uint4*)(kc + ksl[i] * (HKV * D) + h * D + 16 * (t & 3));                       \
            _Pragma("unroll") for (int i = 0; i < 4; ++i)                                                      \
                vr[i] = *(const uint2*)(vc + vsl[i] * (HKV * D) + h * D + 8 * vdd);                            \
        }                                                                                                      \
    }
    VSW_LOADS
    if (spec >= 0) {
        const int real = seqs[b];
        if (real != seq) { seq = real; VSW_LOADS }
    }
#undef VSW_LOADS
    if (abl == 1) {
        unsigned z = kr[0].x ^ kr[1].y ^ vr[0].x ^ vr[1].y ^ vr[2].x ^ vr[3].y;
        asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
        if (z == 12345u) out[tid] = __float2bfloat16(1.f);
        return;
    }

    // Queries: RMS norm, RoPE at seq + token (pairs 16 half + i), FP8 into SW64 smem.
    uint8_t* qs = sm.q[wg];
    {
        float x[32], ss = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const unsigned u[4] = {qraw[j].x, qraw[j].y, qraw[j].z, qraw[j].w};
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                x[8 * j + 2 * k] = bf(u[k] & 0xffff); x[8 * j + 2 * k + 1] = bf(u[k] >> 16);
            }
        }
#pragma unroll
        for (int i = 0; i < 32; ++i) ss += x[i] * x[i];
        ss += __shfl_xor_sync(0xffffffff, ss, 1);
        const float inv = rsqrtf(ss / D + 1e-5f);
        const float cc[16] = {cq[0].x, cq[0].y, cq[0].z, cq[0].w, cq[1].x, cq[1].y, cq[1].z, cq[1].w,
                              cq[2].x, cq[2].y, cq[2].z, cq[2].w, cq[3].x, cq[3].y, cq[3].z, cq[3].w};
        const float sn[16] = {sq[0].x, sq[0].y, sq[0].z, sq[0].w, sq[1].x, sq[1].y, sq[1].z, sq[1].w,
                              sq[2].x, sq[2].y, sq[2].z, sq[2].w, sq[3].x, sq[3].y, sq[3].z, sq[3].w};
        float y[32];
#pragma unroll
        for (int p = 0; p < 16; ++p) {
            y[2 * p] = (x[2 * p] * cc[p] - x[2 * p + 1] * sn[p]) * inv;
            y[2 * p + 1] = (x[2 * p + 1] * cc[p] + x[2 * p] * sn[p]) * inv;
        }
        uint32_t pk[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) pk[j] = pack_e4(y[4 * j], y[4 * j + 1], y[4 * j + 2], y[4 * j + 3]);
        *(uint4*)(qs + swz64(qr * 64 + half * 32)) = make_uint4(pk[0], pk[1], pk[2], pk[3]);
        *(uint4*)(qs + swz64(qr * 64 + half * 32 + 16)) = make_uint4(pk[4], pk[5], pk[6], pk[7]);
    }
    uint8_t* kt = sm.k[wg];
    uint8_t* vt = sm.vt[wg];
    if (!newer) {
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int u = t + 128 * i;
            *(uint4*)(kt + swz64((u >> 2) * 64 + 16 * (u & 3))) = kr[i];
        }
        // V^T: word (d, slots 4 c' .. 4 c' + 3 of key group 16 (t >> 5)) = keys vk0 + {0, 1, 8, 9} at d.
        const uint32_t slot0 = 16 * (t >> 5) + 4 * ((t >> 3) & 3);
#pragma unroll
        for (int hw = 0; hw < 2; ++hw) {
            const uint32_t a = hw ? vr[0].y : vr[0].x, bb = hw ? vr[1].y : vr[1].x;
            const uint32_t cc = hw ? vr[2].y : vr[2].x, dd = hw ? vr[3].y : vr[3].x;
            const uint32_t ablo = prmt(a, bb, 0x5140), abhi = prmt(a, bb, 0x7362);
            const uint32_t cdlo = prmt(cc, dd, 0x5140), cdhi = prmt(cc, dd, 0x7362);
            const uint32_t o4[4] = {prmt(ablo, cdlo, 0x5410), prmt(ablo, cdlo, 0x7632), prmt(abhi, cdhi, 0x5410),
                                    prmt(abhi, cdhi, 0x7632)};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint32_t d = 8 * vdd + 4 * hw + j;
                *(uint32_t*)(vt + swz64(d * 64 + slot0)) = o4[j];
            }
        }
    } else {
        // New tokens: zero V^T (only slots of keys 0..7 get data), then normalize / RoPE / quantize.
        *(uint4*)(vt + 32 * t) = make_uint4(0, 0, 0, 0);
        *(uint4*)(vt + 32 * t + 16) = make_uint4(0, 0, 0, 0);
        float k[4] = {bf(nkw.x & 0xffff), bf(nkw.x >> 16), bf(nkw.y & 0xffff), bf(nkw.y >> 16)};
        float s = k[0] * k[0] + k[1] * k[1] + k[2] * k[2] + k[3] * k[3];
#pragma unroll
        for (int o = 8; o; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
        const float ki = rsqrtf(s / D + 1e-5f) / kscale;
        const float r0 = (k[0] * ncs[0] - k[1] * ncs[2]) * ki, r1 = (k[1] * ncs[0] + k[0] * ncs[2]) * ki;
        const float r2 = (k[2] * ncs[1] - k[3] * ncs[3]) * ki, r3 = (k[3] * ncs[1] + k[2] * ncs[3]) * ki;
        const unsigned k8 = pack_e4(r0, r1, r2, r3);
        const float vi = 1.f / vscale;
        const unsigned v8 = pack_e4(bf(nvw.x & 0xffff) * vi, bf(nvw.x >> 16) * vi, bf(nvw.y & 0xffff) * vi,
                                    bf(nvw.y >> 16) * vi);
        const long sl = nloc * (HKV * D) + h * D + nd;
        *(unsigned*)(kc + sl) = k8;
        *(unsigned*)(vc + sl) = v8;
        *(uint32_t*)(kt + swz64(nt * 64 + nd)) = k8;
        asm volatile("bar.sync 3, 128;" ::: "memory");
        const uint32_t sl0 = 4 * (nt >> 1) + (nt & 1);
#pragma unroll
        for (int j = 0; j < 4; ++j) vt[swz64((nd + j) * 64 + sl0)] = (uint8_t)(v8 >> (8 * j));
    }
    STAMP(1)
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("bar.sync %0, 128;" ::"r"(1 + wg) : "memory"); STAMP(2)

    const float scale = qk_scale * kscale;
    const uint32_t qa = smem_u32(qs), ka = smem_u32(kt), va = smem_u32(vt);
    const int jA = 2 * w, jB = 2 * w + 1;  // tokens of rows 16 w + g and 16 w + 8 + g
    float o[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) o[i] = 0.f;
    float mA, mB, lA = 0.f, lB = 0.f;
    if (!newer) {
        float s[32];
        wg_fence();
        qk_k32(s, mdesc(qa, 512, 2), mdesc(ka, 512, 2), 0);
        qk_k32(s, mdesc(qa + 32, 512, 2), mdesc(ka + 32, 512, 2), 1);
        wg_commit();
        wg_wait<0>();
        STAMP(3)
        fence_regs(s);
        float xa = -INFINITY, xb = -INFINITY;
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int wk = w0 + 8 * j + 2 * c + e;
                const bool ok = wk < W - 1 && base + wk >= 0;
                s[4 * j + e] = ok && wk >= jA ? s[4 * j + e] * scale : -INFINITY;
                s[4 * j + 2 + e] = ok && wk >= jB ? s[4 * j + 2 + e] * scale : -INFINITY;
                xa = fmaxf(xa, s[4 * j + e]);
                xb = fmaxf(xb, s[4 * j + 2 + e]);
            }
        xa = fmaxf(xa, __shfl_xor_sync(0xffffffff, xa, 1));
        xb = fmaxf(xb, __shfl_xor_sync(0xffffffff, xb, 1));
        xa = fmaxf(xa, __shfl_xor_sync(0xffffffff, xa, 2));
        xb = fmaxf(xb, __shfl_xor_sync(0xffffffff, xb, 2));
        mA = xa == -INFINITY ? -1e30f : xa - 8.f;  // FA3's FP8 max offset: P <= 2^8
        mB = xb == -INFINITY ? -1e30f : xb - 8.f;
        float e[32];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            e[4 * j + 0] = ex2(s[4 * j + 0] - mA); e[4 * j + 1] = ex2(s[4 * j + 1] - mA);
            e[4 * j + 2] = ex2(s[4 * j + 2] - mB); e[4 * j + 3] = ex2(s[4 * j + 3] - mB);
            lA += e[4 * j + 0] + e[4 * j + 1]; lB += e[4 * j + 2] + e[4 * j + 3];
        }
        uint32_t p[8];
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            const int b0 = 16 * kk;
            p[kk * 4 + 0] = pack_e4(e[b0 + 0], e[b0 + 1], e[b0 + 4], e[b0 + 5]);
            p[kk * 4 + 1] = pack_e4(e[b0 + 2], e[b0 + 3], e[b0 + 6], e[b0 + 7]);
            p[kk * 4 + 2] = pack_e4(e[b0 + 8], e[b0 + 9], e[b0 + 12], e[b0 + 13]);
            p[kk * 4 + 3] = pack_e4(e[b0 + 10], e[b0 + 11], e[b0 + 14], e[b0 + 15]);
        }
        wg_fence();
        pv8_k32(o, p[0], p[1], p[2], p[3], mdesc(va, 512, 2));
        pv8_k32(o, p[4], p[5], p[6], p[7], mdesc(va + 32, 512, 2));
        wg_commit();
        wg_wait<0>();
        STAMP(3)
        fence_regs(o);
    } else {
        float s[4] = {0.f, 0.f, 0.f, 0.f};
        wg_fence();
        qk8_k32(s, mdesc(qa, 512, 2), mdesc(ka, 512, 2), 0);
        qk8_k32(s, mdesc(qa + 32, 512, 2), mdesc(ka + 32, 512, 2), 1);
        wg_commit();
        wg_wait<0>();
        STAMP(4)
        fence_regs(s);
        float xa = -INFINITY, xb = -INFINITY;
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int i = 2 * c + e;
            s[e] = i <= jA ? s[e] * scale : -INFINITY;
            s[2 + e] = i <= jB ? s[2 + e] * scale : -INFINITY;
            xa = fmaxf(xa, s[e]);
            xb = fmaxf(xb, s[2 + e]);
        }
        xa = fmaxf(xa, __shfl_xor_sync(0xffffffff, xa, 1));
        xb = fmaxf(xb, __shfl_xor_sync(0xffffffff, xb, 1));
        xa = fmaxf(xa, __shfl_xor_sync(0xffffffff, xa, 2));
        xb = fmaxf(xb, __shfl_xor_sync(0xffffffff, xb, 2));
        mA = xa - 8.f;  // key j is always visible to token j
        mB = xb - 8.f;
        const float e0 = ex2(s[0] - mA), e1 = ex2(s[1] - mA), e2 = ex2(s[2] - mB), e3 = ex2(s[3] - mB);
        lA = e0 + e1; lB = e2 + e3;
        wg_fence();
        pv8_k32(o, pack_e4(e0, e1, 0.f, 0.f), pack_e4(e2, e3, 0.f, 0.f), 0u, 0u, mdesc(va, 512, 2));
        wg_commit();
        wg_wait<0>();
        STAMP(4)
        fence_regs(o);
    }
    lA += __shfl_xor_sync(0xffffffff, lA, 1); lA += __shfl_xor_sync(0xffffffff, lA, 2);
    lB += __shfl_xor_sync(0xffffffff, lB, 1); lB += __shfl_xor_sync(0xffffffff, lB, 2);
    if (abl == 2 || abl == 4) {
        asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
        if (abl == 4) asm volatile("barrier.cluster.arrive.aligned;\n\tbarrier.cluster.wait.aligned;" ::: "memory");
        float z = lA + lB;
#pragma unroll
        for (int i = 0; i < 32; ++i) z += o[i];
        if (z == 12345.f) out[tid] = __float2bfloat16(z);
        return;
    }

    // Warpgroups 1 (and 2 in CTA 15) hand their partials to warpgroup 0 (same register layout), which merges.
    const int nthr = rank == 15 ? 384 : 256;
    if (wg > 0) {
        float* x = &sm.xf[wg - 1][0][0];
#pragma unroll
        for (int i = 0; i < 32; ++i) x[i * 128 + t] = o[i];
        x[32 * 128 + t] = mA; x[33 * 128 + t] = lA; x[34 * 128 + t] = mB; x[35 * 128 + t] = lB;
        asm volatile("bar.arrive 4, %0;" ::"r"(nthr) : "memory");
        asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
        if (VSW_BULK == 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;\n\tbarrier.cluster.wait.aligned;" ::: "memory");
        return;
    }
    asm volatile("bar.sync 4, %0;" ::"r"(nthr) : "memory");
    STAMP(5)
    {
        const int nx = rank == 15 ? 2 : 1;
        float xm[2][4];
#pragma unroll
        for (int k = 0; k < 2; ++k)
#pragma unroll
            for (int i = 0; i < 4; ++i) xm[k][i] = k < nx ? sm.xf[k][32 + i][t] : (i & 1 ? 0.f : -1e30f);
        const float MA = fmaxf(mA, fmaxf(xm[0][0], xm[1][0])), MB = fmaxf(mB, fmaxf(xm[0][2], xm[1][2]));
        const float a0 = ex2(mA - MA), b0 = ex2(mB - MB);
        float aw[2], bw[2];
#pragma unroll
        for (int k = 0; k < 2; ++k) { aw[k] = ex2(xm[k][0] - MA); bw[k] = ex2(xm[k][2] - MB); }
        lA = a0 * lA + aw[0] * xm[0][1] + aw[1] * xm[1][1];
        lB = b0 * lB + bw[0] * xm[0][3] + bw[1] * xm[1][3];
        mA = MA; mB = MB;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const bool ra = (i & 2) == 0;
            float v = o[i] * (ra ? a0 : b0) + sm.xf[0][i][t] * (ra ? aw[0] : bw[0]);
            if (nx == 2) v += sm.xf[1][i][t] * (ra ? aw[1] : bw[1]);
            o[i] = v;
        }
    }

    STAMP(6)
    // Push: row 16 w + g -> owner 4 w + g / 4, row 16 w + 8 + g -> owner 4 w + 2 + g / 4; owned row g % 4.
    {
        const int pid = rank, ri = g & 3;
        const int oA = 4 * w + (g >> 2), oB = oA + 2;
        if (VSW_BULK == 2) {
            // Through L2: partial [cluster][src][owner][264], one release/acquire cluster barrier, owners ld.cg.
            float* gA = g_scr + ((long)(cid * 16 + pid) * 16 + oA) * (RO * D + 2 * RO);
            float* gB = g_scr + ((long)(cid * 16 + pid) * 16 + oB) * (RO * D + 2 * RO);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                *(float2*)(gA + ri * D + 8 * j + 2 * c) = make_float2(o[4 * j + 0], o[4 * j + 1]);
                *(float2*)(gB + ri * D + 8 * j + 2 * c) = make_float2(o[4 * j + 2], o[4 * j + 3]);
            }
            if (c == 0) {
                *(float2*)(gA + RO * D + 2 * ri) = make_float2(mA, lA);
                *(float2*)(gB + RO * D + 2 * ri) = make_float2(mB, lB);
            }
            asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
            asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
        } else if (VSW_BULK) {
            // Staged per owner in this CTA's shared memory, then one 1056-byte bulk copy (DSMEM) per owner.
            float* sA = &sm.stage[oA][0];
            float* sB = &sm.stage[oB][0];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                *(float2*)(sA + ri * D + 8 * j + 2 * c) = make_float2(o[4 * j + 0], o[4 * j + 1]);
                *(float2*)(sB + ri * D + 8 * j + 2 * c) = make_float2(o[4 * j + 2], o[4 * j + 3]);
            }
            if (c == 0) {
                *(float2*)(sA + RO * D + 2 * ri) = make_float2(mA, lA);
                *(float2*)(sB + RO * D + 2 * ri) = make_float2(mB, lB);
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("bar.sync 1, 128;" ::: "memory");
            asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
            if (t < 16) {
                const unsigned dst = mapa(smem_u32(&sm.racc[pid][0]), t), rmb = mapa(mb, t);

                asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                             ::"r"(dst), "r"(smem_u32(&sm.stage[t][0])), "n"((RO * D + 2 * RO) * 4), "r"(rmb)
                             : "memory");
            }
        } else {
            asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
            const unsigned mbA = mapa(mb, oA), mbB = mapa(mb, oB);
            const unsigned ra = smem_u32(&sm.racc[pid][ri * D + 2 * c]);
            const unsigned dA = mapa(ra, oA), dB = mapa(ra, oB);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                sa2(dA + 32 * j, mbA, o[4 * j + 0], o[4 * j + 1]);
                sa2(dB + 32 * j, mbB, o[4 * j + 2], o[4 * j + 3]);
            }
            if (c == 0) {
                const unsigned rm = smem_u32(&sm.racc[pid][RO * D + 2 * ri]);
                sa2(mapa(rm, oA), mbA, mA, lA);
                sa2(mapa(rm, oB), mbB, mB, lB);
            }
        }
    }

    // Owner: rows R = 4 rank + warp (token R >> 3, head h 8 + (R & 7)), two d per lane.
    if (tid < 32 * RO) {
        const int R = 4 * rank + (tid >> 5), tok = b * L + (R >> 3), head = h * 8 + (R & 7);
        const unsigned gw = *(const unsigned*)(qkv + (long)tok * QKV_W + (HQ + 2 * HKV) * D + head * D + 2 * lane);
        const int q = tid >> 5;
        if (VSW_BULK == 2) {
#pragma unroll
            for (int s = 0; s < NP; ++s) {
                const float* src = g_scr + ((long)(cid * 16 + s) * 16 + rank) * (RO * D + 2 * RO);
                *(float2*)&sm.racc[s][q * D + 2 * lane] = __ldcg((const float2*)(src + q * D + 2 * lane));
                if (lane == 0) *(float2*)&sm.racc[s][RO * D + 2 * q] = __ldcg((const float2*)(src + RO * D + 2 * q));
            }
            __syncwarp();
        } else
        asm volatile("{\n\t.reg .pred p;\n\tW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n\t@!p bra W;\n\t}"
                     ::"r"(mb) : "memory");
        if (abl == 3) { out[(long)tok * HQ * D + head * D + 2 * lane] = __float2bfloat16(sm.racc[5][q * D + lane]); }
        else {
        float M = -1e30f;
#pragma unroll
        for (int s = 0; s < NP; ++s) M = fmaxf(M, sm.racc[s][RO * D + 2 * q]);
        float den = 0.f, o0 = 0.f, o1 = 0.f;
#pragma unroll
        for (int s = 0; s < NP; ++s) {
            const float2 ml = *(const float2*)&sm.racc[s][RO * D + 2 * q];
            const float wt = ex2(ml.x - M);
            den += wt * ml.y;
            const float2 a = *(const float2*)&sm.racc[s][q * D + 2 * lane];
            o0 += wt * a.x; o1 += wt * a.y;
        }
        const float r = vscale / den;
        o0 *= r; o1 *= r;
        float s2 = o0 * o0 + o1 * o1;
#pragma unroll
        for (int o = 16; o; o >>= 1) s2 += __shfl_xor_sync(0xffffffff, s2, o);
        const float ni = rsqrtf(s2 / D + 1e-5f);
        const float g0 = bf(gw & 0xffff), g1 = bf(gw >> 16);
        o0 *= ni / (1.f + __expf(-g0));
        o1 *= ni / (1.f + __expf(-g1));
        *(__nv_bfloat162*)(out + (long)tok * HQ * D + head * D + 2 * lane) = __floats2bfloat162_rn(o0, o1);
        STAMP(7)
        }
    }
    // Bulk copies read this CTA's staging buffer asynchronously: nobody exits before every owner has its data.
    if (VSW_BULK == 1) asm volatile("barrier.cluster.arrive.relaxed.aligned;\n\tbarrier.cluster.wait.aligned;" ::: "memory");
}
}  // namespace vsw

void vsw_dbg(torch::Tensor out) {
    cudaMemcpyFromSymbol(out.data_ptr(), vsw::g_dbg, sizeof(unsigned long long) * 32 * 3 * 8);
}
void vsw_verify(torch::Tensor qkv, torch::Tensor cs, torch::Tensor kc, torch::Tensor vc, torch::Tensor pt,
                torch::Tensor seq, torch::Tensor loc, torch::Tensor ks, torch::Tensor vs, torch::Tensor out, int64_t page,
                double qk_scale, int64_t abl) {
    int batch = seq.size(0), mp = pt.size(1);
    const int spec = (batch == 1 && page == 1 && abl < 10) ? mp - 8 : -1;
    abl %= 10;
    const int smem = (int)sizeof(vsw::Smem) + 1024;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(32 * batch);
    cfg.blockDim = dim3(384);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = c10::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 16; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.attrs = at; cfg.numAttrs = 1;
#define VSW_ARGS (const unsigned short*)qkv.data_ptr(), cs.data_ptr<float>(), (unsigned char*)kc.data_ptr(), \
    (unsigned char*)vc.data_ptr(), pt.data_ptr<int>(), seq.data_ptr<int>(), loc.data_ptr<long>(), ks.data_ptr<float>(), \
    vs.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(), mp, (float)qk_scale, spec, (int)abl
#define VSW_LAUNCH(P) { static bool set = false; \
        if (!set) { cudaFuncSetAttribute(vsw::verify<P>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1); \
                    cudaFuncSetAttribute(vsw::verify<P>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
                    set = true; } \
        cudaLaunchKernelEx(&cfg, vsw::verify<P>, VSW_ARGS); }
    if (page == 1) VSW_LAUNCH(1)
    else { TORCH_CHECK(page == 64, "vsw: page 1 or 64"); VSW_LAUNCH(64) }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

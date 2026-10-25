// Verify (q_len 8) sliding attention as ONE kernel: per request one 16-CTA thread-block cluster.
// CTA rank r = 8 h + p: KV head h, window keys w in [256 p, 256 p + 256) (w = t - (seq - 2047), cached w < 2047);
// CTA p = 7 of each head also attends the block's 8 new tokens (and writes them to the cache).
// 16 warps: row group rg = warp & 3 (rows 16 rg .. 16 rg + 15; row = 8 token + query head), key quarter
// kq = warp >> 2 (64 keys). QK^T and P@V are mma.sync m16n8k16 FP16 with FP32 accumulators (FP8 K/V convert to
// FP16 exactly; Q in FP16, finer than FA3's E4M3; P in FP16, finer than FA3's E4M3). One pass per warp (no online
// rescale), the 4 key quarters merged in shared memory, then each CTA pushes 8 rows (one token) to its owner CTA
// (st.async + mbarrier complete_tx), which merges 8 partials, normalizes, gates and stores BF16.
// Fixed shapes: HQ 16, HKV 2, D 64, window 2048, L 8.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace vsl {
constexpr int HQ = 16, HKV = 2, D = 64, W = 2048, L = 8, QKV_W = (2 * HQ + 2 * HKV) * D;

__device__ __forceinline__ unsigned short pk8(float hi, float lo) {
    unsigned short r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
    return r;
}
__device__ __forceinline__ unsigned h2(float lo, float hi) {
    unsigned r;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}
__device__ __forceinline__ unsigned e2h(unsigned short x) {
    unsigned r;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(x));
    return r;
}
__device__ __forceinline__ unsigned lo16(unsigned x) { return e2h((unsigned short)(x & 0xffff)); }
__device__ __forceinline__ unsigned hi16(unsigned x) { return e2h((unsigned short)(x >> 16)); }
__device__ __forceinline__ void mma(float* d, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                    unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned mapa(unsigned addr, unsigned cta) {
    unsigned r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(cta));
    return r;
}
__device__ __forceinline__ void sa4(unsigned addr, unsigned mbar, float a, float b, float c, float d) {
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1,%2,%3,%4}, [%5];"
                 ::"r"(addr), "f"(a), "f"(b), "f"(c), "f"(d), "r"(mbar) : "memory");
}
__device__ __forceinline__ void sa2(unsigned addr, unsigned mbar, float a, float b) {
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1,%2}, [%3];"
                 ::"r"(addr), "f"(a), "f"(b), "r"(mbar) : "memory");
}
__device__ __forceinline__ uint4 ldu4(const void* p) {
    uint4 r;
    asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
    return r;
}
__device__ __forceinline__ float4 ldf4(const void* p) {
    float4 r;
    asm volatile("ld.global.nc.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(r.x), "=f"(r.y), "=f"(r.z), "=f"(r.w) : "l"(p));
    return r;
}
__device__ __forceinline__ unsigned long long gtime() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ unsigned long long g_dbg[64 * 16 * 8];
#define STAMP(k) if (abl == 9 && lane == 0) g_dbg[(blockIdx.x * 16 + warp) * 8 + (k)] = gtime();
__device__ __forceinline__ float bf(unsigned short x) { return __uint_as_float((unsigned)x << 16); }
__device__ __forceinline__ long slot_of(const int* ptr, int t, int page) {
    return page == 1 ? (long)ptr[t] : (long)ptr[t / page] * page + t % page;
}

struct Smem {
    float part[16][16][D];     // per warp: its 16 rows' unnormalized partial (row-local d order)
    float pml[16][16][2];      // per warp and row: max, sum
    float racc[64][D];  // received: [source part][owned row][d], NPH x RO = 64 rows
    float rml[64][2];
    unsigned char newk[L][D], newv[L][D];
    unsigned long long mbar;
};

template <int PAGE, int NPH>
__global__ void __launch_bounds__(512, 1) verify(const unsigned short* __restrict__ qkv, const float* __restrict__ cs,
                                                 unsigned char* kc, unsigned char* vc, const int* __restrict__ pt,
                                                 const int* __restrict__ seqs, const long* __restrict__ loc,
                                                 const float* __restrict__ ks, const float* __restrict__ vs,
                                                 __nv_bfloat16* __restrict__ out, int MP, float qk_scale, int abl) {
    extern __shared__ __align__(16) unsigned char smraw[];
    Smem& sm = *reinterpret_cast<Smem*>(smraw);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
    const int rg = warp & 3, kq = warp >> 2;
    unsigned rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    // NPH CTAs per KV head (8: one cluster per request; 16: one cluster per KV head), RO rows owned per CTA.
    constexpr int KC = W / NPH, KW = KC / 4, NT = KW / 8, KS = KW / 16, RO = 64 / NPH;
    const int h = NPH == 8 ? rank >> 3 : (blockIdx.x >> 4) & 1, p = NPH == 8 ? rank & 7 : rank;
    const int b = blockIdx.x / (16 * (NPH / 8));
    const unsigned mb = (unsigned)__cvta_generic_to_shared(&sm.mbar);
    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n\tfence.mbarrier_init.release.cluster;" ::"r"(mb) : "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(mb), "r"(NPH * (RO * D * 4 + RO * 8))
                     : "memory");
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    STAMP(0)

    // Queries: rows 16 rg + g (token 2 rg, head g) and 16 rg + 8 + g (token 2 rg + 1), d 16 c .. 16 c + 15.
    const int tok0 = b * L + 2 * rg;
    const unsigned short* qp0 = qkv + (long)tok0 * QKV_W + (h * 8 + g) * D + 16 * c;
    const unsigned short* qp1 = qp0 + QKV_W;
    const uint4 qa0 = ldu4(qp0), qb0 = ldu4(qp0 + 8);
    const uint4 qa1 = ldu4(qp1), qb1 = ldu4(qp1 + 8);
    const float kscale = *ks, vscale = *vs;
    const int seq = seqs[b];
    const bool newer = p == NPH - 1 && kq == 3;  // these 4 warps also run the 8 new tokens
    // New tokens: thread (of the 128 in warps 12..15) handles token nt = ti >> 4, d 4 (ti & 15) .. + 3.
    const int ti = tid - 384, nt = ti >> 4, nd = 4 * (ti & 15);
    uint2 nkw = make_uint2(0, 0), nvw = make_uint2(0, 0);
    long nloc = 0;
    if (newer) {
        const unsigned short* kp = qkv + (long)(b * L + nt) * QKV_W + (HQ + h) * D + nd;
        nkw = *(const uint2*)kp;
        nvw = *(const uint2*)(kp + HKV * D);
        nloc = loc[b * L + nt];
    }
    const float* csr0 = cs + (long)(seq + 2 * rg) * D + 8 * c;
    const float* csr1 = csr0 + D;
    const float4 c00 = ldf4(csr0), c01 = ldf4(csr0 + 4);
    const float4 s00 = ldf4(csr0 + 32), s01 = ldf4(csr0 + 36);
    const float4 c10 = ldf4(csr1), c11 = ldf4(csr1 + 4);
    const float4 s10 = ldf4(csr1 + 32), s11 = ldf4(csr1 + 36);

    // Keep every page-table-independent load above the gathers (in-order issue: loads after them wait a DRAM trip).
    asm volatile("" ::: "memory");
    // Cached keys of this warp: w0 + [0, 64). K: rows w0 + 8 n + g; V: rows w0 + 16 t + {2c, 2c+1, 2c+8, 2c+9}.
    const int base = seq - (W - 1);
    const int w0 = p * KC + kq * KW;
    const int* ptr = pt + (long)b * MP;
    const int tmax = max(seq - 1, 0);
    uint4 kr[NT];
    uint2 vr[4 * KS];
    {
        long ksl[NT], vsl[4 * KS];
#pragma unroll
        for (int n = 0; n < NT; ++n) ksl[n] = slot_of(ptr, min(max(base + w0 + 8 * n + g, 0), tmax), PAGE);
#pragma unroll
        for (int i = 0; i < 4 * KS; ++i) {
            int k = 16 * (i >> 2) + 2 * c + (i & 1) + 8 * ((i >> 1) & 1);
            vsl[i] = slot_of(ptr, min(max(base + w0 + k, 0), tmax), PAGE);
        }
#pragma unroll
        for (int n = 0; n < NT; ++n) kr[n] = *(const uint4*)(kc + ksl[n] * (HKV * D) + h * D + 16 * c);
#pragma unroll
        for (int i = 0; i < 4 * KS; ++i) vr[i] = *(const uint2*)(vc + vsl[i] * (HKV * D) + h * D + 8 * g);
    }

    if (abl == 1) {
        unsigned z = 0;
#pragma unroll
        for (int n = 0; n < NT; ++n) z ^= kr[n].x ^ kr[n].w;
#pragma unroll
        for (int i = 0; i < 4 * KS; ++i) z ^= vr[i].x ^ vr[i].y;
        asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
        if (z == 12345u) out[tid] = __float2bfloat16(1.f);
        return;
    }
    // Q: rmsnorm, RoPE at seq + token, FP16 A fragments (k-step s: d 16c + 4s + {0,1} | {2,3}).
    unsigned qf[4][4];
    {
        const uint4 qa[2] = {qa0, qa1}, qb[2] = {qb0, qb1};
        const float4 cc4[2][2] = {{c00, c01}, {c10, c11}}, ss4[2][2] = {{s00, s01}, {s10, s11}};
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            float x[16];
            unsigned u[8] = {qa[r].x, qa[r].y, qa[r].z, qa[r].w, qb[r].x, qb[r].y, qb[r].z, qb[r].w};
#pragma unroll
            for (int i = 0; i < 8; ++i) { x[2 * i] = bf(u[i] & 0xffff); x[2 * i + 1] = bf(u[i] >> 16); }
            float ss = 0.f;
#pragma unroll
            for (int i = 0; i < 16; ++i) ss += x[i] * x[i];
            ss += __shfl_xor_sync(0xffffffff, ss, 1);
            ss += __shfl_xor_sync(0xffffffff, ss, 2);
            const float inv = rsqrtf(ss / D + 1e-5f);
            float cc[8] = {cc4[r][0].x, cc4[r][0].y, cc4[r][0].z, cc4[r][0].w, cc4[r][1].x, cc4[r][1].y, cc4[r][1].z, cc4[r][1].w};
            float sn[8] = {ss4[r][0].x, ss4[r][0].y, ss4[r][0].z, ss4[r][0].w, ss4[r][1].x, ss4[r][1].y, ss4[r][1].z, ss4[r][1].w};
            float y[16];
#pragma unroll
            for (int q = 0; q < 8; ++q) {
                y[2 * q] = (x[2 * q] * cc[q] - x[2 * q + 1] * sn[q]) * inv;
                y[2 * q + 1] = (x[2 * q + 1] * cc[q] + x[2 * q] * sn[q]) * inv;
            }
#pragma unroll
            for (int s = 0; s < 4; ++s) {
                qf[s][r] = h2(y[4 * s], y[4 * s + 1]);
                qf[s][2 + r] = h2(y[4 * s + 2], y[4 * s + 3]);
            }
        }
    }

    // New tokens (p == 7, warps 12..15): normalize, RoPE, quantize, store to the cache and to shared memory.
    if (newer) {
        float k[4] = {bf(nkw.x & 0xffff), bf(nkw.x >> 16), bf(nkw.y & 0xffff), bf(nkw.y >> 16)};
        float s = k[0] * k[0] + k[1] * k[1] + k[2] * k[2] + k[3] * k[3];
#pragma unroll
        for (int o = 8; o; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
        const float ki = rsqrtf(s / D + 1e-5f) / kscale;
        const float* cn = cs + (long)(seq + nt) * D + nd / 2;
        const float co0 = cn[0], co1 = cn[1], si0 = cn[32], si1 = cn[33];
        const float r0 = (k[0] * co0 - k[1] * si0) * ki, r1 = (k[1] * co0 + k[0] * si0) * ki;
        const float r2 = (k[2] * co1 - k[3] * si1) * ki, r3 = (k[3] * co1 + k[2] * si1) * ki;
        const unsigned k8 = pk8(r1, r0) | ((unsigned)pk8(r3, r2) << 16);
        const float vi = 1.f / vscale;
        const unsigned v8 = pk8(bf(nvw.x >> 16) * vi, bf(nvw.x & 0xffff) * vi) |
                            ((unsigned)pk8(bf(nvw.y >> 16) * vi, bf(nvw.y & 0xffff) * vi) << 16);
        *(unsigned*)&sm.newk[nt][nd] = k8;
        *(unsigned*)&sm.newv[nt][nd] = v8;
        const long sl = nloc * (HKV * D) + h * D + nd;
        *(unsigned*)(kc + sl) = k8;
        *(unsigned*)(vc + sl) = v8;
        asm volatile("bar.sync 1, 128;" ::: "memory");
    }

    // S = Q K^T: n-tile n holds keys 8n + 2c + {0,1} for rows g (e 0,1) and g + 8 (e 2,3); tile 8 = new tokens.
    float sc[NT + 1][4];
#pragma unroll
    for (int n = 0; n < NT; ++n) {
        sc[n][0] = sc[n][1] = sc[n][2] = sc[n][3] = 0.f;
        const unsigned kw[4] = {kr[n].x, kr[n].y, kr[n].z, kr[n].w};
#pragma unroll
        for (int s = 0; s < 4; ++s) mma(sc[n], qf[s][0], qf[s][1], qf[s][2], qf[s][3], lo16(kw[s]), hi16(kw[s]));
    }
    sc[NT][0] = sc[NT][1] = sc[NT][2] = sc[NT][3] = -INFINITY;
    uint2 nv = make_uint2(0, 0);
    if (newer) {
        sc[NT][0] = sc[NT][1] = sc[NT][2] = sc[NT][3] = 0.f;
        const uint4 nk = *(const uint4*)&sm.newk[g][16 * c];
        const unsigned kw[4] = {nk.x, nk.y, nk.z, nk.w};
#pragma unroll
        for (int s = 0; s < 4; ++s) mma(sc[NT], qf[s][0], qf[s][1], qf[s][2], qf[s][3], lo16(kw[s]), hi16(kw[s]));
    }
    STAMP(1)
    const float scale = qk_scale * kscale;
    const int j0 = 2 * rg, j1 = 2 * rg + 1;
    float m0 = -INFINITY, m1 = -INFINITY;
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int w = w0 + 8 * n + 2 * c + e;
            const bool ok = w < W - 1 && base + w >= 0;
            sc[n][e] = ok && w >= j0 ? sc[n][e] * scale : -INFINITY;
            sc[n][e + 2] = ok && w >= j1 ? sc[n][e + 2] * scale : -INFINITY;
            m0 = fmaxf(m0, sc[n][e]);
            m1 = fmaxf(m1, sc[n][e + 2]);
        }
    if (newer) {
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int i = 2 * c + e;
            sc[NT][e] = i <= j0 ? sc[NT][e] * scale : -INFINITY;
            sc[NT][e + 2] = i <= j1 ? sc[NT][e + 2] * scale : -INFINITY;
            m0 = fmaxf(m0, sc[NT][e]);
            m1 = fmaxf(m1, sc[NT][e + 2]);
        }
        nv = *(const uint2*)&sm.newv[2 * c][8 * g];  // rows 2c, 2c + 1 below
    }
    m0 = fmaxf(m0, __shfl_xor_sync(0xffffffff, m0, 1));
    m0 = fmaxf(m0, __shfl_xor_sync(0xffffffff, m0, 2));
    m1 = fmaxf(m1, __shfl_xor_sync(0xffffffff, m1, 1));
    m1 = fmaxf(m1, __shfl_xor_sync(0xffffffff, m1, 2));
    // A row with no visible key (only possible at tiny contexts) keeps m = -inf: use 0 so exp2 gives 0, not NaN.
    const float mo0 = m0 == -INFINITY ? 0.f : m0, mo1 = m1 == -INFINITY ? 0.f : m1;
    float l0 = 0.f, l1 = 0.f;
#pragma unroll
    for (int n = 0; n < NT + 1; ++n) {
        sc[n][0] = exp2f(sc[n][0] - mo0); sc[n][1] = exp2f(sc[n][1] - mo0);
        sc[n][2] = exp2f(sc[n][2] - mo1); sc[n][3] = exp2f(sc[n][3] - mo1);
        l0 += sc[n][0] + sc[n][1];
        l1 += sc[n][2] + sc[n][3];
    }
    l0 += __shfl_xor_sync(0xffffffff, l0, 1);
    l0 += __shfl_xor_sync(0xffffffff, l0, 2);
    l1 += __shfl_xor_sync(0xffffffff, l1, 1);
    l1 += __shfl_xor_sync(0xffffffff, l1, 2);

    // O = P V: k-step t = keys 16t .. 16t + 15; d-tile n column g = d 8g + n, so acc[n][0..1] = row g d 16c + n,
    // 16c + 8 + n; acc[n][2..3] = row g + 8.
    float acc[8][4];
#pragma unroll
    for (int n = 0; n < 8; ++n) acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.f;
#pragma unroll
    for (int t = 0; t < KS; ++t) {
        const unsigned a0 = h2(sc[2 * t][0], sc[2 * t][1]), a1 = h2(sc[2 * t][2], sc[2 * t][3]);
        const unsigned a2 = h2(sc[2 * t + 1][0], sc[2 * t + 1][1]), a3 = h2(sc[2 * t + 1][2], sc[2 * t + 1][3]);
        const uint2 A = vr[4 * t], B = vr[4 * t + 1], C = vr[4 * t + 2], E = vr[4 * t + 3];
        unsigned b0[8], b1[8];
        {
            unsigned x;
            x = __byte_perm(A.x, B.x, 0x5140); b0[0] = lo16(x); b0[1] = hi16(x);
            x = __byte_perm(A.x, B.x, 0x7362); b0[2] = lo16(x); b0[3] = hi16(x);
            x = __byte_perm(A.y, B.y, 0x5140); b0[4] = lo16(x); b0[5] = hi16(x);
            x = __byte_perm(A.y, B.y, 0x7362); b0[6] = lo16(x); b0[7] = hi16(x);
            x = __byte_perm(C.x, E.x, 0x5140); b1[0] = lo16(x); b1[1] = hi16(x);
            x = __byte_perm(C.x, E.x, 0x7362); b1[2] = lo16(x); b1[3] = hi16(x);
            x = __byte_perm(C.y, E.y, 0x5140); b1[4] = lo16(x); b1[5] = hi16(x);
            x = __byte_perm(C.y, E.y, 0x7362); b1[6] = lo16(x); b1[7] = hi16(x);
        }
#pragma unroll
        for (int n = 0; n < 8; ++n) mma(acc[n], a0, a1, a2, a3, b0[n], b1[n]);
    }
    if (newer) {
        // New tokens: keys 0..7 in the k-step's first half (rows 2c, 2c + 1), second half zero.
        const unsigned a0 = h2(sc[NT][0], sc[NT][1]), a1 = h2(sc[NT][2], sc[NT][3]);
        const uint2 B = *(const uint2*)&sm.newv[2 * c + 1][8 * g];
        const uint2 A = nv;
        unsigned b0[8];
        unsigned x;
        x = __byte_perm(A.x, B.x, 0x5140); b0[0] = lo16(x); b0[1] = hi16(x);
        x = __byte_perm(A.x, B.x, 0x7362); b0[2] = lo16(x); b0[3] = hi16(x);
        x = __byte_perm(A.y, B.y, 0x5140); b0[4] = lo16(x); b0[5] = hi16(x);
        x = __byte_perm(A.y, B.y, 0x7362); b0[6] = lo16(x); b0[7] = hi16(x);
#pragma unroll
        for (int n = 0; n < 8; ++n) mma(acc[n], a0, a1, 0u, 0u, b0[n], 0u);
    }

    if (abl == 2) {
        asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
        float z = l0 + l1;
#pragma unroll
        for (int n = 0; n < 8; ++n) z += acc[n][0] + acc[n][1] + acc[n][2] + acc[n][3];
        if (z == 12345.f) out[tid] = __float2bfloat16(z);
        return;
    }
    STAMP(2)
    // Per-warp partials to shared memory (row-local d), then merge the 4 key quarters per row.
    {
        float* r0 = &sm.part[warp][g][16 * c];
        float* r1 = &sm.part[warp][g + 8][16 * c];
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            *(float4*)(r0 + 4 * q) = make_float4(acc[4 * q][0], acc[4 * q + 1][0], acc[4 * q + 2][0], acc[4 * q + 3][0]);
            *(float4*)(r0 + 8 + 4 * q) = make_float4(acc[4 * q][1], acc[4 * q + 1][1], acc[4 * q + 2][1], acc[4 * q + 3][1]);
            *(float4*)(r1 + 4 * q) = make_float4(acc[4 * q][2], acc[4 * q + 1][2], acc[4 * q + 2][2], acc[4 * q + 3][2]);
            *(float4*)(r1 + 8 + 4 * q) = make_float4(acc[4 * q][3], acc[4 * q + 1][3], acc[4 * q + 2][3], acc[4 * q + 3][3]);
        }
        if (c == 0) {
            *(float2*)&sm.pml[warp][g][0] = make_float2(mo0, l0);
            *(float2*)&sm.pml[warp][g + 8][0] = make_float2(mo1, l1);
        }
    }
    __syncthreads();
    {
        // Thread: row R = tid >> 3 (token R >> 3, head R & 7), d 8 (tid & 7) .. + 7.
        const int R = tid >> 3, dc = tid & 7, rgr = R >> 4, rr = R & 15;
        float mk[4], lk[4], M = -INFINITY;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float2 v = *(const float2*)&sm.pml[4 * k + rgr][rr][0];
            mk[k] = v.x; lk[k] = v.y; M = fmaxf(M, v.x);
        }
        float Lsum = 0.f, o[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float w = exp2f(mk[k] - M);
            Lsum += w * lk[k];
            const float4 a = *(const float4*)&sm.part[4 * k + rgr][rr][8 * dc];
            const float4 a2 = *(const float4*)&sm.part[4 * k + rgr][rr][8 * dc + 4];
            o[0] += w * a.x; o[1] += w * a.y; o[2] += w * a.z; o[3] += w * a.w;
            o[4] += w * a2.x; o[5] += w * a2.y; o[6] += w * a2.z; o[7] += w * a2.w;
        }
        STAMP(3)
        asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
        STAMP(4)
        if (abl == 3) { if (o[0] + o[7] + Lsum == 12345.f) out[tid] = __float2bfloat16(1.f); return; }
        const unsigned owner = (NPH == 8 ? h * 8 : 0) + R / RO;
        const unsigned rmb = mapa(mb, owner);
        const unsigned dst = mapa((unsigned)__cvta_generic_to_shared(&sm.racc[p * RO + R % RO][8 * dc]), owner);
        sa4(dst, rmb, o[0], o[1], o[2], o[3]);
        sa4(dst + 16, rmb, o[4], o[5], o[6], o[7]);
        if (dc == 0) sa2(mapa((unsigned)__cvta_generic_to_shared(&sm.rml[p * RO + R % RO][0]), owner), rmb, M, Lsum);
    }

    // Owner: token p, query heads h * 8 + warp (warps 0..7), two d per lane (row-local d order: 16c' + ...).
    if (warp < RO) {
        const int Rw = p * RO + warp, tok = b * L + (Rw >> 3), head = h * 8 + (Rw & 7);
        const unsigned gw = *(const unsigned*)(qkv + (long)tok * QKV_W + (HQ + 2 * HKV) * D + head * D + 2 * lane);
        asm volatile("{\n\t.reg .pred p;\n\tW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n\t@!p bra W;\n\t}"
                     ::"r"(mb) : "memory");
        STAMP(5)
        float M = -INFINITY;
#pragma unroll
        for (int s = 0; s < NPH; ++s) M = fmaxf(M, sm.rml[s * RO + warp][0]);
        float den = 0.f, o0 = 0.f, o1 = 0.f;
#pragma unroll
        for (int s = 0; s < NPH; ++s) {
            const float w = exp2f(sm.rml[s * RO + warp][0] - M);
            den += w * sm.rml[s * RO + warp][1];
            const float2 a = *(const float2*)&sm.racc[s * RO + warp][2 * lane];
            o0 += w * a.x; o1 += w * a.y;
        }
        const float r = vscale / den;
        o0 *= r; o1 *= r;
        float s2 = o0 * o0 + o1 * o1;
#pragma unroll
        for (int o = 16; o; o >>= 1) s2 += __shfl_xor_sync(0xffffffff, s2, o);
        const float ni = rsqrtf(s2 / D + 1e-5f);
        // Stored index i = 2 lane (+1) is d = 16 (i >> 4) + 8 ((i >> 3) & 1) ... : row-local order is
        // [16c + n] = acc[n][0] (d 16c + n), [16c + 8 + n] = acc[n][1] (d 16c + 8 + n): identity.
        const float g0 = bf(gw & 0xffff), g1 = bf(gw >> 16);
        o0 *= ni / (1.f + __expf(-g0));
        o1 *= ni / (1.f + __expf(-g1));
        *(__nv_bfloat162*)(out + (long)tok * HQ * D + head * D + 2 * lane) = __floats2bfloat162_rn(o0, o1);
        STAMP(6)
    }
}
}  // namespace vsl

void vsl_verify(torch::Tensor qkv, torch::Tensor cs, torch::Tensor kc, torch::Tensor vc, torch::Tensor pt,
                torch::Tensor seq, torch::Tensor loc, torch::Tensor ks, torch::Tensor vs, torch::Tensor out, int64_t page,
                double qk_scale, int64_t abl) {
    int nph = abl >= 10 ? 16 : 8;
    if (abl < 19) abl %= 10; else abl = 9;
    int batch = seq.size(0), mp = pt.size(1);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(16 * batch * (nph / 8));
    cfg.blockDim = dim3(512);
    cfg.dynamicSmemBytes = sizeof(vsl::Smem);
    cfg.stream = c10::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 16; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.attrs = at; cfg.numAttrs = 1;
#define VSL_ARGS (const unsigned short*)qkv.data_ptr(), cs.data_ptr<float>(), (unsigned char*)kc.data_ptr(), \
    (unsigned char*)vc.data_ptr(), pt.data_ptr<int>(), seq.data_ptr<int>(), loc.data_ptr<long>(), ks.data_ptr<float>(), \
    vs.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(), mp, (float)qk_scale, (int)abl
#define VSL_LAUNCH2(P, N) { static bool set = false; \
        if (!set) { cudaFuncSetAttribute(vsl::verify<P, N>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1); \
                    cudaFuncSetAttribute(vsl::verify<P, N>, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(vsl::Smem)); \
                    set = true; } \
        cudaLaunchKernelEx(&cfg, vsl::verify<P, N>, VSL_ARGS); }
#define VSL_LAUNCH(P) { if (nph == 8) VSL_LAUNCH2(P, 8) else VSL_LAUNCH2(P, 16) }
    if (page == 1) VSL_LAUNCH(1)
    else { TORCH_CHECK(page == 64, "vsl: page 1 or 64"); VSL_LAUNCH(64) }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void vsl_dbg(torch::Tensor out) {
    cudaMemcpyFromSymbol(out.data_ptr(), vsl::g_dbg, sizeof(unsigned long long) * 64 * 16 * 8);
}

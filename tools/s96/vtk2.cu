// Batch-1-style decode sliding attention as ONE kernel: per (request, KV head) one 16-CTA thread-block cluster
// (s91: a cluster per KV head halves each SM's gathered bytes; H100 paired decode 1/16 sliding -0.58/-0.14 us).
// CTA rank r covers window keys [128 r, 128 r + 128) of its KV head (4 warps, 32 keys each). QK^T and P@V are mma.sync m16n8k32 E4M3 (P in E4M3 with FA3's max offset 8, V^T built in registers
// by byte permutes), FP32 partials pushed by each warp straight into the owning CTA's shared memory
// (st.shared::cluster), one cluster barrier, then CTA r < 8 merges query head 8 h + r (RMSNorm, sigmoid gate, BF16).
// Fixed shapes: HQ 16, HKV 2, D 64, window 2048, L 1.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>

#ifndef CSL_SPEC
#define CSL_SPEC 1
#endif
namespace vtk {
constexpr int L = 8, HQ = 16, HKV = 2, D = 64, W = 2048, QKV_W = (2 * HQ + 2 * HKV) * D;

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
__device__ __forceinline__ unsigned mapa(unsigned addr, unsigned cta) {
    unsigned r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(cta));
    return r;
}
__device__ __forceinline__ void st4(unsigned addr, float a, float b, float c, float d) {
    asm volatile("st.shared::cluster.v4.f32 [%0], {%1,%2,%3,%4};" ::"r"(addr), "f"(a), "f"(b), "f"(c), "f"(d)
                 : "memory");
}
__device__ __forceinline__ void st2(unsigned addr, float a, float b) {
    asm volatile("st.shared::cluster.v2.f32 [%0], {%1,%2};" ::"r"(addr), "f"(a), "f"(b) : "memory");
}
__device__ __forceinline__ void sa4(unsigned addr, unsigned mbar, float a, float b, float c, float d) {
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1,%2,%3,%4}, [%5];"
                 ::"r"(addr), "f"(a), "f"(b), "f"(c), "f"(d), "r"(mbar) : "memory");
}
__device__ __forceinline__ void sa2(unsigned addr, unsigned mbar, float a, float b) {
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1,%2}, [%3];"
                 ::"r"(addr), "f"(a), "f"(b), "r"(mbar) : "memory");
}
__device__ __forceinline__ float bf(unsigned short x) { return __uint_as_float((unsigned)x << 16); }

__device__ __forceinline__ float e4(unsigned v, int byte) {
    __nv_fp8_e4m3 x; x.__x = (v >> (8 * byte)) & 0xff; return float(x);
}

// One 16-CTA cluster per (request, token pair T0 = 2 tp, T0 + 1, KV head): mma rows g = token T0's 8 heads, rows
// g + 8 = token T0 + 1's. Window keys are indexed on token T0 (base = pos0 - 2047); token T0 + 1 drops window key 0
// and adds its own new key (window key 2048) on CUDA cores in the new-token warp.
template <int PAGE>
__global__ void __launch_bounds__(128) verify(const unsigned short* __restrict__ qkv, const float* __restrict__ cs,
                                              unsigned char* kc, unsigned char* vc, const int* __restrict__ pt,
                                              const int* __restrict__ seqs, const long* __restrict__ loc,
                                              const float* __restrict__ ks, const float* __restrict__ vs,
                                              __nv_bfloat16* __restrict__ out, int MP, float qk_scale, int spec) {
    __shared__ __align__(16) float racc[16][D];
    __shared__ __align__(16) float rml[16][2];
    __shared__ __align__(16) float lpart[4][16][D];
    __shared__ __align__(16) float lml[4][16][2];
    __shared__ __align__(16) unsigned char newkv[L][2][D];
    __shared__ __align__(8) unsigned long long mbar;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
    const int ci = blockIdx.x >> 4, h = ci & 1, tp = ci >> 1, b = tp / (L / 2), T0 = 2 * (tp % (L / 2)), wk = warp;
    const int tokA = b * L + T0;
    unsigned rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    const unsigned mb = (unsigned)__cvta_generic_to_shared(&mbar);
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n\tfence.mbarrier_init.release.cluster;" ::"r"(mb) : "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(mb), "r"(16 * (16 * 16 + 8)) : "memory");
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

    const unsigned short* qp = qkv + (long)tokA * QKV_W + (h * 8 + g) * D + 16 * c;
    const uint4 qa = *(const uint4*)qp, qb = *(const uint4*)(qp + 8);
    const uint4 qc = *(const uint4*)(qp + QKV_W), qd = *(const uint4*)(qp + QKV_W + 8);
    const bool newer = rank == 15 && wk == 3;
    unsigned nkw[L], nvw[L];
    if (newer) {
#pragma unroll
        for (int n = 0; n < L; ++n) {
            if (n <= T0 + 1) {
                const unsigned short* kp = qkv + (long)(b * L + n) * QKV_W + (HQ + h) * D + 2 * lane;
                nkw[n] = *(const unsigned*)kp; nvw[n] = *(const unsigned*)(kp + HKV * D);
            }
        }
    }
    const int otok = tokA + (rank >> 3);
    const unsigned gw = *(const unsigned*)(qkv + (long)otok * QKV_W + (HQ + 2 * HKV) * D + (h * 8 + (rank & 7)) * D + 2 * lane);
    const float kscale = *ks, vscale = *vs;
    int seq = spec >= 0 ? spec : seqs[b];
    int pos;
    const float* csr;
    float4 c0, c1, s0, s1, c2, c3, s2, s3;
    int base, w0, tmax;
    const int* ptr = pt + (long)b * MP;
    long kslot[4], vslot[8];
    uint4 kr[4];
    uint2 vr[8];
#define CSL_LOADS                                                                                              \
    pos = seq + T0;                                                                                            \
    csr = cs + (long)pos * D + 8 * c;                                                                          \
    c0 = *(const float4*)csr; c1 = *(const float4*)(csr + 4);                                                  \
    s0 = *(const float4*)(csr + 32); s1 = *(const float4*)(csr + 36);                                          \
    c2 = *(const float4*)(csr + D); c3 = *(const float4*)(csr + D + 4);                                        \
    s2 = *(const float4*)(csr + D + 32); s3 = *(const float4*)(csr + D + 36);                                  \
    base = pos - (W - 1);                                                                                      \
    w0 = rank * 128 + wk * 32;                                                                                 \
    tmax = max(seq - 1, 0);                                                                                    \
    _Pragma("unroll") for (int j = 0; j < 4; ++j) {                                                            \
        int t = min(max(base + w0 + 8 * j + g, 0), tmax);                                                      \
        kslot[j] = PAGE == 1 ? (long)ptr[t] : (long)ptr[t / PAGE] * PAGE + t % PAGE;                           \
    }                                                                                                          \
    _Pragma("unroll") for (int i = 0; i < 8; ++i) {                                                            \
        int k = 8 * (i >> 1) + 2 * c + (i & 1);                                                                \
        int t = min(max(base + w0 + k, 0), tmax);                                                              \
        vslot[i] = PAGE == 1 ? (long)ptr[t] : (long)ptr[t / PAGE] * PAGE + t % PAGE;                           \
    }                                                                                                          \
    _Pragma("unroll") for (int j = 0; j < 4; ++j) kr[j] = *(const uint4*)(kc + kslot[j] * (HKV * D) + h * D + 16 * c); \
    _Pragma("unroll") for (int i = 0; i < 8; ++i) vr[i] = *(const uint2*)(vc + vslot[i] * (HKV * D) + h * D + 8 * g);
    CSL_LOADS
    if (spec >= 0) {
        const int real = seqs[b];
        if (real != seq) { seq = real; CSL_LOADS }
    }
#undef CSL_LOADS
    const float g0 = bf(gw & 0xffff), g1 = bf(gw >> 16);
    unsigned q8[4], q9[4];
    {
        auto prep = [&](uint4 ua, uint4 ub, float4 a0, float4 a1, float4 b0, float4 b1, unsigned* qo) {
            float x[16];
            unsigned u[8] = {ua.x, ua.y, ua.z, ua.w, ub.x, ub.y, ub.z, ub.w};
#pragma unroll
            for (int i = 0; i < 8; ++i) { x[2 * i] = bf(u[i] & 0xffff); x[2 * i + 1] = bf(u[i] >> 16); }
            float ss = 0.f;
#pragma unroll
            for (int i = 0; i < 16; ++i) ss += x[i] * x[i];
            ss += __shfl_xor_sync(0xffffffff, ss, 1);
            ss += __shfl_xor_sync(0xffffffff, ss, 2);
            const float inv = rsqrtf(ss / D + 1e-5f);
            float cc[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            float sn[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
            float y[16];
#pragma unroll
            for (int p = 0; p < 8; ++p) {
                y[2 * p] = (x[2 * p] * cc[p] - x[2 * p + 1] * sn[p]) * inv;
                y[2 * p + 1] = (x[2 * p + 1] * cc[p] + x[2 * p] * sn[p]) * inv;
            }
#pragma unroll
            for (int w = 0; w < 4; ++w) qo[w] = pk8(y[4 * w + 1], y[4 * w]) | (pk8(y[4 * w + 3], y[4 * w + 2]) << 16);
        };
        prep(qa, qb, c0, c1, s0, s1, q8);
        prep(qc, qd, c2, c3, s2, s3, q9);
    }

    // New tokens n <= T0 + 1 (window keys 2047 - T0 + n): CTA 15's last warp, K tile 3 (keys 24 + g), V rows 24 + 2c + e;
    // n = T0 + 1 is window key 2048 (token T0 + 1 only). This cluster writes tokens T0, T0 + 1's cache rows.
    if (newer) {
#pragma unroll
        for (int n = 0; n < L; ++n) {
            if (n <= T0 + 1) {
                const float co = cs[(long)(seq + n) * D + lane], si = cs[(long)(seq + n) * D + 32 + lane];
                float k0 = bf(nkw[n] & 0xffff), k1 = bf(nkw[n] >> 16);
                float s = k0 * k0 + k1 * k1;
#pragma unroll
                for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
                float ki = rsqrtf(s / D + 1e-5f) / kscale;
                float r0 = (k0 * co - k1 * si) * ki, r1 = (k1 * co + k0 * si) * ki;
                unsigned short k8 = pk8(r1, r0);
                unsigned short v8 = pk8(bf(nvw[n] >> 16) / vscale, bf(nvw[n] & 0xffff) / vscale);
                *(unsigned short*)&newkv[n][0][2 * lane] = k8;
                *(unsigned short*)&newkv[n][1][2 * lane] = v8;
                if (n >= T0) {
                    long slot = loc[b * L + n] * (HKV * D) + h * D + 2 * lane;
                    *(unsigned short*)(kc + slot) = k8;
                    *(unsigned short*)(vc + slot) = v8;
                }
            }
        }
        __syncwarp();
        const int nk = g + T0 - 7;
        if (nk >= 0) kr[3] = *(const uint4*)&newkv[nk][0][16 * c];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int nv = 2 * c + e + T0 - 7;
            if (nv >= 0) vr[6 + e] = *(const uint2*)&newkv[nv][1][8 * g];
        }
    }

    // S = Q K^T: tile j holds keys 8j + 2c + {0, 1}; [0..1] row g (token T0), [2..3] row g + 8 (token T0 + 1).
    float sc[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.f;
        mma8(sc[j], q8[0], q9[0], q8[1], q9[1], kr[j].x, kr[j].y);
        mma8(sc[j], q8[2], q9[2], q8[3], q9[3], kr[j].z, kr[j].w);
    }
    const float scale = qk_scale * kscale;
    // Token T0 + 1's own key (window key 2048), in the new-token warp only.
    float sx = -INFINITY;
    if (newer) {
        const uint4 kn = *(const uint4*)&newkv[T0 + 1][0][16 * c];
        const unsigned kw4[4] = {kn.x, kn.y, kn.z, kn.w};
        float d = 0.f;
#pragma unroll
        for (int w = 0; w < 4; ++w)
#pragma unroll
            for (int y = 0; y < 4; ++y) d += e4(q9[w], y) * e4(kw4[w], y);
        d += __shfl_xor_sync(0xffffffff, d, 1);
        d += __shfl_xor_sync(0xffffffff, d, 2);
        sx = d * scale;
    }
    float ma = -1e30f, mb2 = sx > -1e30f ? sx : -1e30f;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int w = w0 + 8 * j + 2 * c + e;
            bool ok = base + w >= 0;
            sc[j][e] = ok ? sc[j][e] * scale : -INFINITY;
            sc[j][e + 2] = ok && w > 0 ? sc[j][e + 2] * scale : -INFINITY;
            ma = fmaxf(ma, sc[j][e]);
            mb2 = fmaxf(mb2, sc[j][e + 2]);
        }
    ma = fmaxf(ma, __shfl_xor_sync(0xffffffff, ma, 1));
    ma = fmaxf(ma, __shfl_xor_sync(0xffffffff, ma, 2));
    mb2 = fmaxf(mb2, __shfl_xor_sync(0xffffffff, mb2, 1));
    mb2 = fmaxf(mb2, __shfl_xor_sync(0xffffffff, mb2, 2));
    const float moa = ma - 8.f, mob = mb2 - 8.f;  // FA3's FP8 max offset: P <= 2^8
    float la = 0.f, lb = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            sc[j][e] = exp2f(sc[j][e] - moa); la += sc[j][e];
            sc[j][e + 2] = exp2f(sc[j][e + 2] - mob); lb += sc[j][e + 2];
        }
    la += __shfl_xor_sync(0xffffffff, la, 1);
    la += __shfl_xor_sync(0xffffffff, la, 2);
    lb += __shfl_xor_sync(0xffffffff, lb, 1);
    lb += __shfl_xor_sync(0xffffffff, lb, 2);
    const unsigned pa0 = pk8(sc[0][1], sc[0][0]) | (pk8(sc[1][1], sc[1][0]) << 16);
    const unsigned pa1 = pk8(sc[2][1], sc[2][0]) | (pk8(sc[3][1], sc[3][0]) << 16);
    const unsigned pb0 = pk8(sc[0][3], sc[0][2]) | (pk8(sc[1][3], sc[1][2]) << 16);
    const unsigned pb1 = pk8(sc[2][3], sc[2][2]) | (pk8(sc[3][3], sc[3][2]) << 16);

    // O^T = V^T P^T: m-tile i holds d 8g + 2i (m = g) and 8g + 2i + 1 (m = g + 8); n = query rows 2c, 2c + 1.
    float acc[4][4], acb[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int bsel = (i & 1) ? 0x7362 : 0x5140;
        unsigned w[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) w[k] = (i < 2) ? vr[k].x : vr[k].y;
        unsigned t01 = __byte_perm(w[0], w[1], bsel), t23 = __byte_perm(w[2], w[3], bsel);
        unsigned t45 = __byte_perm(w[4], w[5], bsel), t67 = __byte_perm(w[6], w[7], bsel);
        const unsigned v0 = __byte_perm(t01, t23, 0x5410), v1 = __byte_perm(t01, t23, 0x7632);
        const unsigned v2 = __byte_perm(t45, t67, 0x5410), v3 = __byte_perm(t45, t67, 0x7632);
        acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
        acb[i][0] = acb[i][1] = acb[i][2] = acb[i][3] = 0.f;
        mma8(acc[i], v0, v1, v2, v3, pa0, pa1);
        mma8(acb[i], v0, v1, v2, v3, pb0, pb1);
    }
    if (newer) {
        // Key 2048 for token T0 + 1: P in E4M3 like the mma path, FP32 products and sums.
        const float px = exp2f(sx - mob);
        lb += px;
        unsigned short p8 = pk8(0.f, px);
        __nv_fp8_e4m3 pq; pq.__x = p8 & 0xff;
        const float pqf = float(pq);
        const float p0 = __shfl_sync(0xffffffff, pqf, 8 * c), p1 = __shfl_sync(0xffffffff, pqf, 8 * c + 4);
        const uint2 vn = *(const uint2*)&newkv[T0 + 1][1][8 * g];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned vw = i < 2 ? vn.x : vn.y;
            const float va = e4(vw, 2 * (i & 1)), vb = e4(vw, 2 * (i & 1) + 1);
            acb[i][0] += p0 * va; acb[i][1] += p1 * va; acb[i][2] += p0 * vb; acb[i][3] += p1 * vb;
        }
    }

    {
        float* lp = &lpart[wk][0][0];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            float* r = lp + (2 * c + e) * D + 8 * g;
            *(float4*)r = make_float4(acc[0][e], acc[0][e + 2], acc[1][e], acc[1][e + 2]);
            *(float4*)(r + 4) = make_float4(acc[2][e], acc[2][e + 2], acc[3][e], acc[3][e + 2]);
            r += 8 * D;
            *(float4*)r = make_float4(acb[0][e], acb[0][e + 2], acb[1][e], acb[1][e + 2]);
            *(float4*)(r + 4) = make_float4(acb[2][e], acb[2][e + 2], acb[3][e], acb[3][e + 2]);
        }
        if (c == 0) {
            *(float2*)&lml[wk][g][0] = make_float2(moa, la);
            *(float2*)&lml[wk][8 + g][0] = make_float2(mob, lb);
        }
    }
    __syncthreads();
    {
        const int row = tid >> 3, dc = tid & 7;
        float mk[4], lk[4], M = -1e30f;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            float2 v = *(const float2*)&lml[k][row][0];
            mk[k] = v.x; lk[k] = v.y; M = fmaxf(M, v.x);
        }
        float L2 = 0.f, o[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            float w = exp2f(mk[k] - M);
            L2 += w * lk[k];
            float4 a = *(const float4*)&lpart[k][row][8 * dc], a2 = *(const float4*)&lpart[k][row][8 * dc + 4];
            o[0] += w * a.x; o[1] += w * a.y; o[2] += w * a.z; o[3] += w * a.w;
            o[4] += w * a2.x; o[5] += w * a2.y; o[6] += w * a2.z; o[7] += w * a2.w;
        }
        asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
        const unsigned owner = row;
        const unsigned rmb = mapa(mb, owner);
        sa4(mapa((unsigned)__cvta_generic_to_shared(&racc[rank][8 * dc]), owner), rmb, o[0], o[1], o[2], o[3]);
        sa4(mapa((unsigned)__cvta_generic_to_shared(&racc[rank][8 * dc + 4]), owner), rmb, o[4], o[5], o[6], o[7]);
        if (dc == 0) sa2(mapa((unsigned)__cvta_generic_to_shared(&rml[rank][0]), owner), rmb, M, L2);
    }

    // Merge row `rank` (token T0 + rank / 8, query head 8 h + rank % 8) from its 16 partials: one warp.
    if (warp == 0) {
        asm volatile("{\n\t.reg .pred p;\n\tW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n\t@!p bra W;\n\t}"
                     ::"r"(mb) : "memory");
        float M = -1e30f;
#pragma unroll
        for (int s = 0; s < 16; ++s) M = fmaxf(M, rml[s][0]);
        float den = 0.f, o0 = 0.f, o1 = 0.f;
#pragma unroll
        for (int s = 0; s < 16; ++s) {
            float w = exp2f(rml[s][0] - M);
            den += w * rml[s][1];
            float2 a = *(const float2*)&racc[s][2 * lane];
            o0 += w * a.x; o1 += w * a.y;
        }
        float r = vscale / den;
        o0 *= r; o1 *= r;
        float s2 = o0 * o0 + o1 * o1;
#pragma unroll
        for (int o = 16; o; o >>= 1) s2 += __shfl_xor_sync(0xffffffff, s2, o);
        float ni = rsqrtf(s2 / D + 1e-5f);
        o0 *= ni / (1.f + __expf(-g0));
        o1 *= ni / (1.f + __expf(-g1));
        *(__nv_bfloat162*)(out + (long)otok * HQ * D + (h * 8 + (rank & 7)) * D + 2 * lane) = __floats2bfloat162_rn(o0, o1);
    }
}
}  // namespace vtk

void vtk_verify(torch::Tensor qkv, torch::Tensor cs, torch::Tensor kc, torch::Tensor vc, torch::Tensor pt,
                torch::Tensor seq, torch::Tensor loc, torch::Tensor ks, torch::Tensor vs, torch::Tensor out, int64_t page,
                double qk_scale, int64_t usespec, int64_t ntok) {
    int batch = seq.size(0), mp = pt.size(1);
    const int spec = (batch == 1 && page == 1 && usespec) ? mp - vtk::L : -1;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(ntok > 0 ? 32 * ntok : 16 * batch * vtk::L);
    cfg.blockDim = dim3(128);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = c10::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 16; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.attrs = at; cfg.numAttrs = 1;
#define VTK_ARGS (const unsigned short*)qkv.data_ptr(), cs.data_ptr<float>(), (unsigned char*)kc.data_ptr(), \
    (unsigned char*)vc.data_ptr(), pt.data_ptr<int>(), seq.data_ptr<int>(), loc.data_ptr<long>(), ks.data_ptr<float>(), \
    vs.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(), mp, (float)qk_scale, spec
    if (page == 1) {
        static bool set = false;
        if (!set) { cudaFuncSetAttribute(vtk::verify<1>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1); set = true; }
        cudaLaunchKernelEx(&cfg, vtk::verify<1>, VTK_ARGS);
    } else {
        static bool set = false;
        if (!set) { cudaFuncSetAttribute(vtk::verify<64>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1); set = true; }
        TORCH_CHECK(page == 64, "vtk: page 1 or 64");
        cudaLaunchKernelEx(&cfg, vtk::verify<64>, VTK_ARGS);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

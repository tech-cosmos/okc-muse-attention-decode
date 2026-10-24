// Decode sliding attention as ONE kernel, batch-32 variant of csl.cu: per request one 8-CTA cluster.
// CTA rank r covers window keys [256 r, 256 r + 256) for both KV heads; warp (h = warp / 4, wk = warp & 3) takes
// chunk A = keys 256 r + 32 wk + [0, 32) gathered straight into registers (as csl.cu) and chunk B = chunk A + 128
// gathered with cp.async into shared memory (no registers held while in flight). One joint max over both chunks.
// mma.sync m16n8k32 E4M3 for QK^T and P@V (P in E4M3 with FA3's max offset 8). The 4 warps of a head pre-merge
// in smem; each (head, row) partial is pushed with st.async + mbarrier complete_tx to its owner CTA (query heads
// 2r, 2r + 1 live in CTA r), and warps 0 / 1 merge one query head each (RMSNorm, sigmoid gate, BF16).
// Fixed shapes: HQ 16, HKV 2, D 64, window 2048, L 1.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#ifndef ONL
#define ONL 1
#endif
#ifndef ABL
#define ABL 0
#endif
namespace csl8 {
constexpr int HQ = 16, HKV = 2, D = 64, W = 2048, QKV_W = (2 * HQ + 2 * HKV) * D, NC = 8, VST = 80;

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
__device__ __forceinline__ void sa4(unsigned addr, unsigned mbar, float a, float b, float c, float d) {
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 [%0], {%1,%2,%3,%4}, [%5];"
                 ::"r"(addr), "f"(a), "f"(b), "f"(c), "f"(d), "r"(mbar) : "memory");
}
__device__ __forceinline__ void sa2(unsigned addr, unsigned mbar, float a, float b) {
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1,%2}, [%3];"
                 ::"r"(addr), "f"(a), "f"(b), "r"(mbar) : "memory");
}
__device__ __forceinline__ void cpa16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src)
                 : "memory");
}
__device__ __forceinline__ float bf(unsigned short x) { return __uint_as_float((unsigned)x << 16); }

// V^T P^T for one 32-key chunk: vr[k] = 8 bytes (d 8g..8g+7) of key rows {2c, 2c+1, 8+2c, 9+2c, ...}.
__device__ __forceinline__ void pv(float (*acc)[4], const uint2* vr, unsigned pb0, unsigned pb1) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int bsel = (i & 1) ? 0x7362 : 0x5140;
        unsigned w[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) w[k] = (i < 2) ? vr[k].x : vr[k].y;
        unsigned t01 = __byte_perm(w[0], w[1], bsel), t23 = __byte_perm(w[2], w[3], bsel);
        unsigned t45 = __byte_perm(w[4], w[5], bsel), t67 = __byte_perm(w[6], w[7], bsel);
        mma8(acc[i], __byte_perm(t01, t23, 0x5410), __byte_perm(t01, t23, 0x7632), __byte_perm(t45, t67, 0x5410),
             __byte_perm(t45, t67, 0x7632), pb0, pb1);
    }
}

template <int PAGE>
__global__ void __launch_bounds__(256) decode(const unsigned short* __restrict__ qkv, const float* __restrict__ cs,
                                              unsigned char* kc, unsigned char* vc, const int* __restrict__ pt,
                                              const int* __restrict__ seqs, const long* __restrict__ loc,
                                              const float* __restrict__ ks, const float* __restrict__ vs,
                                              __nv_bfloat16* __restrict__ out, int MP, float qk_scale) {
    __shared__ __align__(16) unsigned char sk[8][32][D];
    __shared__ __align__(16) unsigned char sv[8][32][VST];
    __shared__ __align__(16) float racc[2][NC][D];
    __shared__ __align__(16) float rml[2][NC][2];
    __shared__ __align__(16) float lml[2][4][8][2];
    __shared__ __align__(8) unsigned long long mbar;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
    const int h = warp >> 2, wk = warp & 3;
    unsigned rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    const int b = blockIdx.x / NC;
    const unsigned mb = (unsigned)__cvta_generic_to_shared(&mbar);
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n\tfence.mbarrier_init.release.cluster;" ::"r"(mb) : "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(mb), "r"(2 * NC * (D * 4 + 8)) : "memory");
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

    // Page-table-independent loads first (in-order issue), then cos/sin, then the gathers.
    const unsigned short* qp = qkv + (long)b * QKV_W + (h * 8 + g) * D + 16 * c;
    const uint4 qa = *(const uint4*)qp, qb = *(const uint4*)(qp + 8);
    const unsigned short* kp = qkv + (long)b * QKV_W + (HQ + h) * D + 2 * lane;
    const unsigned kw = *(const unsigned*)kp, vw = *(const unsigned*)(kp + HKV * D);
    const unsigned gw = *(const unsigned*)(qkv + (long)b * QKV_W + (HQ + 2 * HKV) * D + (2 * rank + (warp & 1)) * D + 2 * lane);
    const long nloc = loc[b];
    const float kscale = *ks, vscale = *vs;
    const int seq = seqs[b];
    const float* csr = cs + (long)seq * D + 8 * c;
    const float4 c0 = *(const float4*)csr, c1 = *(const float4*)(csr + 4);
    const float4 s0 = *(const float4*)(csr + 32), s1 = *(const float4*)(csr + 36);
    const float co = cs[(long)seq * D + lane], si = cs[(long)seq * D + 32 + lane];
    const int base = seq - (W - 1);
    const int w0 = rank * 256 + wk * 32, w1 = w0 + 128;
    const int* ptr = pt + (long)b * MP;
    const int tmax = max(seq - 1, 0);
    long kslot[4], vslot[8], bslot[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        int t = min(max(base + w0 + 8 * j + g, 0), tmax);
        kslot[j] = PAGE == 1 ? (long)ptr[t] : (long)ptr[t / PAGE] * PAGE + t % PAGE;
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int k = 8 * (i >> 1) + 2 * c + (i & 1);
        int t = min(max(base + w0 + k, 0), tmax);
        vslot[i] = PAGE == 1 ? (long)ptr[t] : (long)ptr[t / PAGE] * PAGE + t % PAGE;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        int t = min(max(base + w1 + 8 * j + g, 0), tmax);
        bslot[j] = PAGE == 1 ? (long)ptr[t] : (long)ptr[t / PAGE] * PAGE + t % PAGE;
    }
    uint4 kr[4];
    uint2 vr[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) kr[j] = *(const uint4*)(kc + kslot[j] * (HKV * D) + h * D + 16 * c);
#pragma unroll
    for (int i = 0; i < 8; ++i) vr[i] = *(const uint2*)(vc + vslot[i] * (HKV * D) + h * D + 8 * g);
    // Chunk B: row 8j + g, 16-byte part c of this head's K and V.
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        cpa16(&sk[warp][8 * j + g][16 * c], kc + bslot[j] * (HKV * D) + h * D + 16 * c);
        cpa16(&sv[warp][8 * j + g][16 * c], vc + bslot[j] * (HKV * D) + h * D + 16 * c);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");

#if ABL == 1
    {
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        unsigned z = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) z ^= kr[j].x ^ kr[j].w ^ *(const unsigned*)&sk[warp][8 * j + g][16 * c];
#pragma unroll
        for (int i = 0; i < 8; ++i) z ^= vr[i].x ^ vr[i].y;
        asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
        if (z == 12345u) out[tid] = __float2bfloat16(1.f);
        return;
    }
#endif
    const float g0 = bf(gw & 0xffff), g1 = bf(gw >> 16);
    float x[16];
    {
        unsigned u[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) { x[2 * i] = bf(u[i] & 0xffff); x[2 * i + 1] = bf(u[i] >> 16); }
    }
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) ss += x[i] * x[i];
    ss += __shfl_xor_sync(0xffffffff, ss, 1);
    ss += __shfl_xor_sync(0xffffffff, ss, 2);
    const float inv = rsqrtf(ss / D + 1e-5f);
    unsigned q8[4];
    {
        float cc[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
        float sn[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
        float y[16];
#pragma unroll
        for (int p = 0; p < 8; ++p) {
            y[2 * p] = (x[2 * p] * cc[p] - x[2 * p + 1] * sn[p]) * inv;
            y[2 * p + 1] = (x[2 * p + 1] * cc[p] + x[2 * p] * sn[p]) * inv;
        }
#pragma unroll
        for (int w = 0; w < 4; ++w) q8[w] = pk8(y[4 * w + 1], y[4 * w]) | (pk8(y[4 * w + 3], y[4 * w + 2]) << 16);
    }

    // The new token (window key 2047 = chunk B's last key of CTA 7, wk 3).
    const bool newer = rank == NC - 1 && wk == 3;
    unsigned short k8 = 0, v8 = 0;
    if (newer) {
        float k0 = bf(kw & 0xffff), k1 = bf(kw >> 16);
        float s = k0 * k0 + k1 * k1;
#pragma unroll
        for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
        float ki = rsqrtf(s / D + 1e-5f) / kscale;
        float r0 = (k0 * co - k1 * si) * ki, r1 = (k1 * co + k0 * si) * ki;
        k8 = pk8(r1, r0);
        v8 = pk8(bf(vw >> 16) / vscale, bf(vw & 0xffff) / vscale);
        long slot = nloc * (HKV * D) + h * D + 2 * lane;
        *(unsigned short*)(kc + slot) = k8;
        *(unsigned short*)(vc + slot) = v8;
    }

    // S = Q K^T for row g: tile j holds keys 8j + 2c + {0, 1}; chunk A from registers.
    float sc[8][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.f;
        mma8(sc[j], q8[0], 0u, q8[1], 0u, kr[j].x, kr[j].y);
        mma8(sc[j], q8[2], 0u, q8[3], 0u, kr[j].z, kr[j].w);
    }
#if ONL
    // Chunk A's softmax and P@V while chunk B is still in flight, then B with an online rescale.
    const float scale = qk_scale * kscale;
    float ma = -1e30f;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            bool ok = base + w0 + 8 * j + 2 * c + e >= 0;
            sc[j][e] = ok ? sc[j][e] * scale : -INFINITY;
            ma = fmaxf(ma, sc[j][e]);
        }
    ma = fmaxf(ma, __shfl_xor_sync(0xffffffff, ma, 1));
    ma = fmaxf(ma, __shfl_xor_sync(0xffffffff, ma, 2));
    const float moa = ma - 8.f;  // FA3's FP8 max offset: P <= 2^8
    float l = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int e = 0; e < 2; ++e) { sc[j][e] = exp2f(sc[j][e] - moa); l += sc[j][e]; }
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
    pv(acc, vr, pk8(sc[0][1], sc[0][0]) | (pk8(sc[1][1], sc[1][0]) << 16),
       pk8(sc[2][1], sc[2][0]) | (pk8(sc[3][1], sc[3][0]) << 16));
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    if (newer) {
        *(unsigned short*)&sk[warp][31][2 * lane] = k8;
        *(unsigned short*)&sv[warp][31][2 * lane] = v8;
    }
    __syncwarp();
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint4 kb = *(const uint4*)&sk[warp][8 * j + g][16 * c];
        float* s = sc[4 + j];
        s[0] = s[1] = s[2] = s[3] = 0.f;
        mma8(s, q8[0], 0u, q8[1], 0u, kb.x, kb.y);
        mma8(s, q8[2], 0u, q8[3], 0u, kb.z, kb.w);
    }
    float m = ma;
#pragma unroll
    for (int j = 4; j < 8; ++j)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            bool ok = base + w1 + 8 * (j & 3) + 2 * c + e >= 0;
            sc[j][e] = ok ? sc[j][e] * scale : -INFINITY;
            m = fmaxf(m, sc[j][e]);
        }
    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1));
    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 2));
    const float mo = m - 8.f;
    const float alpha = exp2f(moa - mo);
    l *= alpha;
#pragma unroll
    for (int j = 4; j < 8; ++j)
#pragma unroll
        for (int e = 0; e < 2; ++e) { sc[j][e] = exp2f(sc[j][e] - mo); l += sc[j][e]; }
    l += __shfl_xor_sync(0xffffffff, l, 1);
    l += __shfl_xor_sync(0xffffffff, l, 2);
    {
        // acc[i][0, 2] hold query row 2c, acc[i][1, 3] row 2c + 1; alpha of row r lives in lanes 4r..4r+3.
        const float a0 = __shfl_sync(0xffffffff, alpha, 8 * c), a1 = __shfl_sync(0xffffffff, alpha, 8 * c + 4);
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[i][0] *= a0; acc[i][2] *= a0; acc[i][1] *= a1; acc[i][3] *= a1; }
        uint2 vb[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) vb[i] = *(const uint2*)&sv[warp][8 * (i >> 1) + 2 * c + (i & 1)][8 * g];
        pv(acc, vb, pk8(sc[4][1], sc[4][0]) | (pk8(sc[5][1], sc[5][0]) << 16),
           pk8(sc[6][1], sc[6][0]) | (pk8(sc[7][1], sc[7][0]) << 16));
    }
#else
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    if (newer) {
        *(unsigned short*)&sk[warp][31][2 * lane] = k8;
        *(unsigned short*)&sv[warp][31][2 * lane] = v8;
    }
    __syncwarp();
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const uint4 kb = *(const uint4*)&sk[warp][8 * j + g][16 * c];
        float* s = sc[4 + j];
        s[0] = s[1] = s[2] = s[3] = 0.f;
        mma8(s, q8[0], 0u, q8[1], 0u, kb.x, kb.y);
        mma8(s, q8[2], 0u, q8[3], 0u, kb.z, kb.w);
    }
    const float scale = qk_scale * kscale;
    float m = -1e30f;
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            bool ok = base + (j < 4 ? w0 : w1) + 8 * (j & 3) + 2 * c + e >= 0;
            sc[j][e] = ok ? sc[j][e] * scale : -INFINITY;
            m = fmaxf(m, sc[j][e]);
        }
    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1));
    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 2));
    const float mo = m - 8.f;  // FA3's FP8 max offset: P <= 2^8
    float l = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int e = 0; e < 2; ++e) { sc[j][e] = exp2f(sc[j][e] - mo); l += sc[j][e]; }
    l += __shfl_xor_sync(0xffffffff, l, 1);
    l += __shfl_xor_sync(0xffffffff, l, 2);

    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
    pv(acc, vr, pk8(sc[0][1], sc[0][0]) | (pk8(sc[1][1], sc[1][0]) << 16),
       pk8(sc[2][1], sc[2][0]) | (pk8(sc[3][1], sc[3][0]) << 16));
    {
        uint2 vb[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) vb[i] = *(const uint2*)&sv[warp][8 * (i >> 1) + 2 * c + (i & 1)][8 * g];
        pv(acc, vb, pk8(sc[4][1], sc[4][0]) | (pk8(sc[5][1], sc[5][0]) << 16),
           pk8(sc[6][1], sc[6][0]) | (pk8(sc[7][1], sc[7][0]) << 16));
    }
#endif

    // Pre-merge the 4 warps of each head: warp w's partial (8 rows x 64 floats) reuses its own sk[w] buffer.
    float (*lpart)[4][8][D] = reinterpret_cast<float (*)[4][8][D]>(&sk[0][0][0]);
    __syncwarp();
    {
        float* lp = &lpart[h][wk][0][0];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            float* r = lp + (2 * c + e) * D + 8 * g;
            *(float4*)r = make_float4(acc[0][e], acc[0][e + 2], acc[1][e], acc[1][e + 2]);
            *(float4*)(r + 4) = make_float4(acc[2][e], acc[2][e + 2], acc[3][e], acc[3][e + 2]);
        }
        if (c == 0) *(float2*)&lml[h][wk][g][0] = make_float2(mo, l);
    }
#if ABL == 2
    {
        asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
        float z = acc[0][0] + acc[1][1] + acc[2][2] + acc[3][3] + l;
        if (z == 12345.f) out[tid] = __float2bfloat16(1.f);
        return;
    }
#endif
    __syncthreads();
    {
        const int h2 = tid >> 7, row = (tid >> 4) & 7, dc = tid & 15;
        float mk[4], lk[4], M = -1e30f;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            float2 v = *(const float2*)&lml[h2][k][row][0];
            mk[k] = v.x; lk[k] = v.y; M = fmaxf(M, v.x);
        }
        float L = 0.f, o[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            float w = exp2f(mk[k] - M);
            L += w * lk[k];
            float4 a = *(const float4*)&lpart[h2][k][row][4 * dc];
            o[0] += w * a.x; o[1] += w * a.y; o[2] += w * a.z; o[3] += w * a.w;
        }
        asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
        const unsigned qh = h2 * 8 + row, owner = qh >> 1, slot = qh & 1;
        const unsigned rmb = mapa(mb, owner);
        sa4(mapa((unsigned)__cvta_generic_to_shared(&racc[slot][rank][4 * dc]), owner), rmb, o[0], o[1], o[2], o[3]);
        if (dc == 0) sa2(mapa((unsigned)__cvta_generic_to_shared(&rml[slot][rank][0]), owner), rmb, M, L);
    }

    // Merge query head 2 rank + warp from its 8 partials: one warp, two d per lane.
    if (warp < 2) {
        asm volatile("{\n\t.reg .pred p;\n\tW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n\t@!p bra W;\n\t}"
                     ::"r"(mb) : "memory");
        float M = -1e30f;
#pragma unroll
        for (int s = 0; s < NC; ++s) M = fmaxf(M, rml[warp][s][0]);
        float den = 0.f, o0 = 0.f, o1 = 0.f;
#pragma unroll
        for (int s = 0; s < NC; ++s) {
            float w = exp2f(rml[warp][s][0] - M);
            den += w * rml[warp][s][1];
            float2 a = *(const float2*)&racc[warp][s][2 * lane];
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
        *(__nv_bfloat162*)(out + (long)b * HQ * D + (2 * rank + warp) * D + 2 * lane) = __floats2bfloat162_rn(o0, o1);
    }
}
}  // namespace csl8

void csl8_decode(torch::Tensor qkv, torch::Tensor cs, torch::Tensor kc, torch::Tensor vc, torch::Tensor pt,
                 torch::Tensor seq, torch::Tensor loc, torch::Tensor ks, torch::Tensor vs, torch::Tensor out,
                 int64_t page, double qk_scale) {
    int batch = seq.size(0), mp = pt.size(1);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(csl8::NC * batch);
    cfg.blockDim = dim3(256);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = c10::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = csl8::NC; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.attrs = at; cfg.numAttrs = 1;
#define CSL8_ARGS (const unsigned short*)qkv.data_ptr(), cs.data_ptr<float>(), (unsigned char*)kc.data_ptr(), \
    (unsigned char*)vc.data_ptr(), pt.data_ptr<int>(), seq.data_ptr<int>(), loc.data_ptr<long>(), ks.data_ptr<float>(), \
    vs.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(), mp, (float)qk_scale
    TORCH_CHECK(page == 1 || page == 64, "csl8: page 1 or 64");
    if (page == 1) cudaLaunchKernelEx(&cfg, csl8::decode<1>, CSL8_ARGS);
    else cudaLaunchKernelEx(&cfg, csl8::decode<64>, CSL8_ARGS);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

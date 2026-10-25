template <int PAGE>
__global__ void __launch_bounds__(32) decode(const unsigned short* __restrict__ qkv, const float* __restrict__ cs,
                                             unsigned char* kc, unsigned char* vc, const int* __restrict__ pt,
                                             const int* __restrict__ seqs, const long* __restrict__ loc,
                                             const float* __restrict__ ks, const float* __restrict__ vs,
                                             __nv_bfloat16* __restrict__ out, int MP, float qk_scale, int spec) {
    // s109 w1: one warp per CTA covering the CTA's 128 keys as 4 chunks of 32 (csl's per-warp layout), so the
    // CTA partial is the warp's own: no shared-memory premerge, no __syncthreads; pushes go straight to owners.
    __shared__ __align__(16) float racc[16][D];
    __shared__ __align__(16) float rml[16][2];
    __shared__ __align__(16) unsigned char newkv[2][2][D];
    __shared__ __align__(8) unsigned long long mbar;
    constexpr int NC = 4;

    const int lane = threadIdx.x, g = lane >> 2, c = lane & 3;
    const int h = (blockIdx.x >> 4) & 1;
    unsigned rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    const int b = blockIdx.x / 32;
    const unsigned mb = (unsigned)__cvta_generic_to_shared(&mbar);
    if (lane == 0 && rank < 8) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n\tfence.mbarrier_init.release.cluster;" ::"r"(mb) : "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(mb), "r"(16 * (16 * 16 + 8)) : "memory");
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

    const unsigned short* qp = qkv + (long)b * QKV_W + (h * 8 + g) * D + 16 * c;
    const uint4 qa = *(const uint4*)qp, qb = *(const uint4*)(qp + 8);
    const unsigned short* kp = qkv + (long)b * QKV_W + (HQ + h) * D + 2 * lane;
    const unsigned kw = *(const unsigned*)kp, vw = *(const unsigned*)(kp + HKV * D);
    const unsigned gw = *(const unsigned*)(qkv + (long)b * QKV_W + (HQ + 2 * HKV) * D + (h * 8 + (rank & 7)) * D + 2 * lane);
    const long nloc = loc[b];
    const float kscale = *ks, vscale = *vs;
    int seq = spec >= 0 ? spec : seqs[b];
    const float* csr;
    float4 c0, c1, s0, s1;
    float co, si;
    int base, w0, tmax;
    const int* ptr = pt + (long)b * MP;
    long kslot[4 * NC], vslot[8 * NC];
    uint4 kr[4 * NC];
    uint2 vr[8 * NC];
#define CSL_LOADS                                                                                              \
    csr = cs + (long)seq * D + 8 * c;                                                                          \
    c0 = *(const float4*)csr; c1 = *(const float4*)(csr + 4);                                                  \
    s0 = *(const float4*)(csr + 32); s1 = *(const float4*)(csr + 36);                                          \
    co = cs[(long)seq * D + lane]; si = cs[(long)seq * D + 32 + lane];                                         \
    base = seq - (W - 1);                                                                                      \
    w0 = rank * 128;                                                                                           \
    tmax = max(seq - 1, 0);                                                                                    \
    _Pragma("unroll") for (int j = 0; j < 4 * NC; ++j) {                                                       \
        int t = min(max(base + w0 + 8 * j + g, 0), tmax);                                                      \
        kslot[j] = PAGE == 1 ? (long)max(ptr[t], 0) : (long)max(ptr[t / PAGE], 0) * PAGE + t % PAGE;           \
    }                                                                                                          \
    _Pragma("unroll") for (int i = 0; i < 8 * NC; ++i) {                                                       \
        int k = 8 * (i >> 1) + 2 * c + (i & 1);                                                                \
        int t = min(max(base + w0 + k, 0), tmax);                                                              \
        vslot[i] = PAGE == 1 ? (long)max(ptr[t], 0) : (long)max(ptr[t / PAGE], 0) * PAGE + t % PAGE;           \
    }                                                                                                          \
    _Pragma("unroll") for (int j = 0; j < 4 * NC; ++j) kr[j] = *(const uint4*)(kc + kslot[j] * (HKV * D) + h * D + 16 * c); \
    _Pragma("unroll") for (int i = 0; i < 8 * NC; ++i) vr[i] = *(const uint2*)(vc + vslot[i] * (HKV * D) + h * D + 8 * g);
    CSL_LOADS
    if (spec >= 0) {
        const int real = seqs[b];
        if (real != seq) { seq = real; CSL_LOADS }
    }
#undef CSL_LOADS
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

    // The new token (window key 2047): CTA 15, the last chunk.
    if (rank == 15) {
        float k0 = bf(kw & 0xffff), k1 = bf(kw >> 16);
        float s = k0 * k0 + k1 * k1;
#pragma unroll
        for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
        float ki = rsqrtf(s / D + 1e-5f) / kscale;
        float r0 = (k0 * co - k1 * si) * ki, r1 = (k1 * co + k0 * si) * ki;
        unsigned short k8 = pk8(r1, r0);
        unsigned short v8 = pk8(bf(vw >> 16) / vscale, bf(vw & 0xffff) / vscale);
        *(unsigned short*)&newkv[h][0][2 * lane] = k8;
        *(unsigned short*)&newkv[h][1][2 * lane] = v8;
        long slot = nloc * (HKV * D) + h * D + 2 * lane;
        *(unsigned short*)(kc + slot) = k8;
        *(unsigned short*)(vc + slot) = v8;
        __syncwarp();
        if (g == 7) kr[4 * NC - 1] = *(const uint4*)&newkv[h][0][16 * c];
        if (c == 3) vr[8 * NC - 1] = *(const uint2*)&newkv[h][1][8 * g];
    }

    float sc[4 * NC][4];
#pragma unroll
    for (int j = 0; j < 4 * NC; ++j) {
        sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.f;
        mma8(sc[j], q8[0], 0u, q8[1], 0u, kr[j].x, kr[j].y);
        mma8(sc[j], q8[2], 0u, q8[3], 0u, kr[j].z, kr[j].w);
    }
    const float scale = qk_scale * kscale;
    float m = -1e30f;
#pragma unroll
    for (int j = 0; j < 4 * NC; ++j)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            bool ok = base + w0 + 8 * j + 2 * c + e >= 0;
            sc[j][e] = ok ? sc[j][e] * scale : -INFINITY;
            m = fmaxf(m, sc[j][e]);
        }
    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1));
    m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 2));
    const float mo = m - 8.f;  // FA3's FP8 max offset: P <= 2^8
    float l = 0.f;
#pragma unroll
    for (int j = 0; j < 4 * NC; ++j)
#pragma unroll
        for (int e = 0; e < 2; ++e) { sc[j][e] = exp2f(sc[j][e] - mo); l += sc[j][e]; }
    l += __shfl_xor_sync(0xffffffff, l, 1);
    l += __shfl_xor_sync(0xffffffff, l, 2);

    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
#pragma unroll
    for (int ch = 0; ch < NC; ++ch) {
        const unsigned pb0 = pk8(sc[4 * ch][1], sc[4 * ch][0]) | (pk8(sc[4 * ch + 1][1], sc[4 * ch + 1][0]) << 16);
        const unsigned pb1 = pk8(sc[4 * ch + 2][1], sc[4 * ch + 2][0]) | (pk8(sc[4 * ch + 3][1], sc[4 * ch + 3][0]) << 16);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int bsel = (i & 1) ? 0x7362 : 0x5140;
            unsigned w[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) w[k] = (i < 2) ? vr[8 * ch + k].x : vr[8 * ch + k].y;
            unsigned t01 = __byte_perm(w[0], w[1], bsel), t23 = __byte_perm(w[2], w[3], bsel);
            unsigned t45 = __byte_perm(w[4], w[5], bsel), t67 = __byte_perm(w[6], w[7], bsel);
            mma8(acc[i], __byte_perm(t01, t23, 0x5410), __byte_perm(t01, t23, 0x7632), __byte_perm(t45, t67, 0x5410),
                 __byte_perm(t45, t67, 0x7632), pb0, pb1);
        }
    }

    // Push this CTA's partial straight to the owners: lane (g, c) holds rows 2c + e, d 8g .. 8g + 7.
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
#pragma unroll
    for (int e = 0; e < 2; ++e) {
        const unsigned owner = 2 * c + e;
        const unsigned rmb = mapa(mb, owner);
        const unsigned dst = mapa((unsigned)__cvta_generic_to_shared(&racc[rank][8 * g]), owner);
        sa4(dst, rmb, acc[0][e], acc[0][e + 2], acc[1][e], acc[1][e + 2]);
        sa4(dst + 16, rmb, acc[2][e], acc[2][e + 2], acc[3][e], acc[3][e + 2]);
    }
    if (c == 0) sa2(mapa((unsigned)__cvta_generic_to_shared(&rml[rank][0]), g), mapa(mb, g), mo, l);

    if (rank < 8) {
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
        *(__nv_bfloat162*)(out + (long)b * HQ * D + (h * 8 + rank) * D + 2 * lane) = __floats2bfloat162_rn(o0, o1);
    }
}

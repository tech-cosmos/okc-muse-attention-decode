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

#ifndef ABL
#define ABL 0
#endif
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

template <int PAGE>
__global__ void __launch_bounds__(128) verify(const unsigned short* __restrict__ qkv, const float* __restrict__ cs,
                                              unsigned char* kc, unsigned char* vc, const int* __restrict__ pt,
                                              const int* __restrict__ seqs, const long* __restrict__ loc,
                                              const float* __restrict__ ks, const float* __restrict__ vs,
                                              __nv_bfloat16* __restrict__ out, int MP, float qk_scale, int spec) {
    __shared__ __align__(16) float racc[16][D];
    __shared__ __align__(16) float rml[16][2];
    __shared__ __align__(16) float lpart[2][4][8][D];
    __shared__ __align__(16) float lml[2][4][8][2];
    __shared__ __align__(8) unsigned long long mbar;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
    const int h = (blockIdx.x >> 4) & 1, wk = warp;
    const int tok = blockIdx.x >> 5, b = tok / L, jt = tok % L;  // this cluster's query token (row of qkv / out)
    unsigned rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    // The owner's receive barrier counts the 16 partials' bytes (st.async complete_tx). Peers must have started
    // (and initialized it) before anyone pushes: arrive now, wait before the pushes.
    const unsigned mb = (unsigned)__cvta_generic_to_shared(&mbar);
    if (threadIdx.x == 0 && rank < 8) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n\tfence.mbarrier_init.release.cluster;" ::"r"(mb) : "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(mb), "r"(16 * (16 * 16 + 8)) : "memory");
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

    // Every load that doesn't need the page table goes out first (warps issue in order, and the gathers' addresses
    // wait for the page-table trip): queries, the new token, the gate, scales; then cos/sin (needs seq).
    const unsigned short* qp = qkv + (long)tok * QKV_W + (h * 8 + g) * D + 16 * c;
    const uint4 qa = *(const uint4*)qp, qb = *(const uint4*)(qp + 8);
    // The block's new tokens 0..jt (only the new-token warp needs them).
    const bool newer = rank == 15 && wk == 3;
    const unsigned gw = *(const unsigned*)(qkv + (long)tok * QKV_W + (HQ + 2 * HKV) * D + (h * 8 + (rank & 7)) * D + 2 * lane);
    const long nloc = loc[tok];
    const float kscale = *ks, vscale = *vs;
    const float rks = 1.f / kscale, rvs = 1.f / vscale;
    // spec >= 0 (batch 1, page 1): SGLang slices the page table to the batch's longest length, so the window's
    // page-table range is known from the table width: gather from it without waiting for seq_lens, then compare
    // with the real length and redo the loads if it differs (never on SGLang's inputs; correct either way).
    int seq = spec >= 0 ? spec : seqs[b];
    // The block's new tokens 0..jt (only the new-token warp needs them), cos/sin rows and cache rows, all issued
    // before the gathers (clamped indices so the loads are unconditional).
    unsigned nkw[L], nvw[L];
    float nco[L], nsi[L];
#pragma unroll
    for (int n = 0; n < L; ++n) {
        // Predicated (not branched) so they stay in the first basic block, issued with the gathers.
        const int nn = min(n, jt);
        const unsigned short* kp = qkv + (long)(b * L + nn) * QKV_W + (HQ + h) * D + 2 * lane;
        nkw[n] = newer ? *(const unsigned*)kp : 0u; nvw[n] = newer ? *(const unsigned*)(kp + HKV * D) : 0u;
        nco[n] = newer ? cs[(long)(seq + nn) * D + lane] : 0.f; nsi[n] = newer ? cs[(long)(seq + nn) * D + 32 + lane] : 0.f;
    }
    int pos;
    const float* csr;
    float4 c0, c1, s0, s1;
    int base, w0, tmax;
    const int* ptr = pt + (long)b * MP;
    long kslot[4], vslot[8];
    uint4 kr[4];
    uint2 vr[8];
#define CSL_LOADS                                                                                              \
    pos = seq + jt;                                                                                            \
    csr = cs + (long)pos * D + 8 * c;                                                                          \
    c0 = *(const float4*)csr; c1 = *(const float4*)(csr + 4);                                                  \
    s0 = *(const float4*)(csr + 32); s1 = *(const float4*)(csr + 36);                                          \
    base = pos - (W - 1) - (ABL == 9 ? jt * 2100 : 0);                                                                                \
    w0 = rank * 128 + wk * 32;                                                                                 \
    tmax = max(seq - 1, 0);                                                                                    \
    /* Keys of this thread: K rows 8j + g (j 0..3), V rows {2c, 2c+1, 8+2c, 9+2c, 16+2c, ...}. */             \
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
        if (real != seq) {
            seq = real; CSL_LOADS
            if (newer) {
#pragma unroll
                for (int n = 0; n < L; ++n) { nco[n] = cs[(long)(seq + n) * D + lane]; nsi[n] = cs[(long)(seq + n) * D + 32 + lane]; }
            }
        }
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

    // The new tokens n <= jt sit at window keys 2047 - jt + n: all in CTA 15's last warp, K tile 3 (keys 24 + g),
    // V rows 24 + 2c + e. Cluster jt writes token jt's cache row for its head.
    if (newer) {
        __shared__ __align__(16) unsigned char newkv[L][2][D];
#pragma unroll
        unsigned short kj = 0, vj = 0;
#pragma unroll
        for (int n = 0; n < L; ++n) {
            {  // branch-free: n > jt repeats token jt (same values, same smem slot)
                const float co = nco[n], si = nsi[n];
                float k0 = bf(nkw[n] & 0xffff), k1 = bf(nkw[n] >> 16);
                float s = k0 * k0 + k1 * k1;
#pragma unroll
                for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
                float ki = rsqrtf(s / D + 1e-5f) * rks;
                float r0 = (k0 * co - k1 * si) * ki, r1 = (k1 * co + k0 * si) * ki;
                unsigned short k8 = pk8(r1, r0);
                unsigned short v8 = pk8(bf(nvw[n] >> 16) * rvs, bf(nvw[n] & 0xffff) * rvs);
                *(unsigned short*)&newkv[n][0][2 * lane] = k8;  // slots n > jt are never read
                *(unsigned short*)&newkv[n][1][2 * lane] = v8;
                kj = n == jt ? k8 : kj; vj = n == jt ? v8 : vj;
            }
        }
        {
            long slot = nloc * (HKV * D) + h * D + 2 * lane;
            *(unsigned short*)(kc + slot) = kj;
            *(unsigned short*)(vc + slot) = vj;
        }
        __syncwarp();
        const int nk = g + jt - 7;
        if (nk >= 0) kr[3] = *(const uint4*)&newkv[nk][0][16 * c];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int nv = 2 * c + e + jt - 7;
            if (nv >= 0) vr[6 + e] = *(const uint2*)&newkv[nv][1][8 * g];
        }
    }

    // S = Q K^T for row g: tile j holds keys 8j + 2c + {0, 1}.
    float sc[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.f;
        mma8(sc[j], q8[0], 0u, q8[1], 0u, kr[j].x, kr[j].y);
        mma8(sc[j], q8[2], 0u, q8[3], 0u, kr[j].z, kr[j].w);
    }
    const float scale = qk_scale * kscale;
    float m = -1e30f;
#pragma unroll
    for (int j = 0; j < 4; ++j)
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
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int e = 0; e < 2; ++e) { sc[j][e] = exp2f(sc[j][e] - mo); l += sc[j][e]; }
    l += __shfl_xor_sync(0xffffffff, l, 1);
    l += __shfl_xor_sync(0xffffffff, l, 2);
    const unsigned pb0 = pk8(sc[0][1], sc[0][0]) | (pk8(sc[1][1], sc[1][0]) << 16);
    const unsigned pb1 = pk8(sc[2][1], sc[2][0]) | (pk8(sc[3][1], sc[3][0]) << 16);

    // O^T = V^T P^T: m-tile i holds d 8g + 2i (rows m = g) and 8g + 2i + 1 (m = g + 8).
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int bsel = (i & 1) ? 0x7362 : 0x5140;
        unsigned w[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) w[k] = (i < 2) ? vr[k].x : vr[k].y;
        unsigned t01 = __byte_perm(w[0], w[1], bsel), t23 = __byte_perm(w[2], w[3], bsel);
        unsigned t45 = __byte_perm(w[4], w[5], bsel), t67 = __byte_perm(w[6], w[7], bsel);
        acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
        mma8(acc[i], __byte_perm(t01, t23, 0x5410), __byte_perm(t01, t23, 0x7632), __byte_perm(t45, t67, 0x5410),
             __byte_perm(t45, t67, 0x7632), pb0, pb1);
    }

    // Pre-merge the 4 warps of each KV head in shared memory, then push one partial per (head, row) to its owner.
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
    __syncthreads();
    {
        const int h2 = h, row = (tid >> 4) & 7, dc = tid & 15;
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
        const unsigned owner = row;
        const unsigned rmb = mapa(mb, owner);
        sa4(mapa((unsigned)__cvta_generic_to_shared(&racc[rank][4 * dc]), owner), rmb, o[0], o[1], o[2], o[3]);
        if (dc == 0) sa2(mapa((unsigned)__cvta_generic_to_shared(&rml[rank][0]), owner), rmb, M, L);
    }

    // Merge query head `rank` from its 16 partials: one warp, two d per lane, no block barriers.
    if (warp == 0 && rank < 8) {
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
        *(__nv_bfloat162*)(out + (long)tok * HQ * D + (h * 8 + rank) * D + 2 * lane) = __floats2bfloat162_rn(o0, o1);
    }
}
}  // namespace vtk

void vtk_verify(torch::Tensor qkv, torch::Tensor cs, torch::Tensor kc, torch::Tensor vc, torch::Tensor pt,
                torch::Tensor seq, torch::Tensor loc, torch::Tensor ks, torch::Tensor vs, torch::Tensor out, int64_t page,
                double qk_scale, int64_t usespec, int64_t ntok) {
    int batch = seq.size(0), mp = pt.size(1);
    const int spec = (batch == 1 && page == 1 && usespec) ? mp - vtk::L : -1;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(ntok > 0 ? 32 * ntok : 32 * batch * vtk::L);
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

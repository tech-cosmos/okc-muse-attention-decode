// Batch-1-style decode sliding attention as ONE kernel: per request one 16-CTA thread-block cluster.
// CTA rank r covers window keys [128 r, 128 r + 128) for both KV heads (8 warps: head = warp / 4, 32 keys
// each). QK^T and P@V are mma.sync m16n8k32 E4M3 (P in E4M3 with FA3's max offset 8, V^T built in registers
// by byte permutes), FP32 partials pushed by each warp straight into the owning CTA's shared memory
// (st.shared::cluster), one cluster barrier, then CTA r merges query head r (RMSNorm, sigmoid gate, BF16).
// Fixed shapes: HQ 16, HKV 2, D 64, window 2048, L 1.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace csla {
constexpr int HQ = 16, HKV = 2, D = 64, W = 2048, QKV_W = (2 * HQ + 2 * HKV) * D;

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
#ifndef MINB
#define MINB 1
#endif
#ifndef ASYNC
#define ASYNC 1
#endif
__global__ void __launch_bounds__(256, MINB) decode(const unsigned short* __restrict__ qkv, const float* __restrict__ cs,
                                              unsigned char* kc, unsigned char* vc, const int* __restrict__ pt,
                                              const int* __restrict__ seqs, const long* __restrict__ loc,
                                              const float* __restrict__ ks, const float* __restrict__ vs,
                                              __nv_bfloat16* __restrict__ out, int MP, float qk_scale) {
    __shared__ __align__(16) float racc[16][D];
    __shared__ __align__(16) float rml[16][2];
    __shared__ __align__(16) float lpart[2][4][8][D];
    __shared__ __align__(16) float lml[2][4][8][2];
    __shared__ __align__(16) unsigned char newkv[2][2][D];
    __shared__ __align__(8) unsigned long long mbar;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
    const int h = warp >> 2, wk = warp & 3;
    unsigned rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    const int b = blockIdx.x / 16;
    // The owner's receive barrier counts the 16 partials' bytes (st.async complete_tx). Peers must have started
    // (and initialized it) before anyone pushes: arrive now, wait before the pushes.
    const unsigned mb = (unsigned)__cvta_generic_to_shared(&mbar);
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n\tfence.mbarrier_init.release.cluster;" ::"r"(mb) : "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(mb), "r"(16 * (16 * 16 + 8)) : "memory");
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

    // Every load that doesn't need the page table goes out first (warps issue in order, and the gathers' addresses
    // wait for the page-table trip): queries, the new token, the gate, scales; then cos/sin (needs seq).
    const unsigned short* qp = qkv + (long)b * QKV_W + (h * 8 + g) * D + 16 * c;
    const uint4 qa = *(const uint4*)qp, qb = *(const uint4*)(qp + 8);
    const unsigned short* kp = qkv + (long)b * QKV_W + (HQ + h) * D + 2 * lane;
    const unsigned kw = *(const unsigned*)kp, vw = *(const unsigned*)(kp + HKV * D);
    const unsigned gw = *(const unsigned*)(qkv + (long)b * QKV_W + (HQ + 2 * HKV) * D + rank * D + 2 * lane);
    const long nloc = loc[b];
    const float kscale = *ks, vscale = *vs;
    const int seq = seqs[b];
    const float* csr = cs + (long)seq * D + 8 * c;
    const float4 c0 = *(const float4*)csr, c1 = *(const float4*)(csr + 4);
    const float4 s0 = *(const float4*)(csr + 32), s1 = *(const float4*)(csr + 36);
    const float co = cs[(long)seq * D + lane], si = cs[(long)seq * D + 32 + lane];
    const int base = seq - (W - 1);
    const int w0 = rank * 128 + wk * 32;
    const int* ptr = pt + (long)b * MP;
    const int tmax = max(seq - 1, 0);
    // Keys of this thread: K rows 8j + g (j 0..3), V rows {2c, 2c+1, 8+2c, 9+2c, 16+2c, ...}.
    long kslot[4], vslot[8];
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
    uint4 kr[4];
    uint2 vr[8];
#if ASYNC
    // Gathers into this thread's own shared-memory slots (no registers held while they fly; read back by the
    // same thread after wait_group, so no block barrier).
    extern __shared__ __align__(16) unsigned char dsm[];
    uint4 (*skr)[256] = reinterpret_cast<uint4 (*)[256]>(dsm);
    uint2 (*svr)[256] = reinterpret_cast<uint2 (*)[256]>(dsm + 4 * 256 * 16);
#pragma unroll
    for (int j = 0; j < 4; ++j)
        if (ASYNC & 1) asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"((unsigned)__cvta_generic_to_shared(&skr[j][tid])),
                     "l"(kc + kslot[j] * (HKV * D) + h * D + 16 * c) : "memory");
        else kr[j] = *(const uint4*)(kc + kslot[j] * (HKV * D) + h * D + 16 * c);
#pragma unroll
    for (int i = 0; i < 8; ++i)
        if (ASYNC & 2) asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"((unsigned)__cvta_generic_to_shared(&svr[i][tid])),
                     "l"(vc + vslot[i] * (HKV * D) + h * D + 8 * g) : "memory");
        else vr[i] = *(const uint2*)(vc + vslot[i] * (HKV * D) + h * D + 8 * g);
    asm volatile("cp.async.commit_group;" ::: "memory");
#else
#pragma unroll
    for (int j = 0; j < 4; ++j) kr[j] = *(const uint4*)(kc + kslot[j] * (HKV * D) + h * D + 16 * c);
#pragma unroll
    for (int i = 0; i < 8; ++i) vr[i] = *(const uint2*)(vc + vslot[i] * (HKV * D) + h * D + 8 * g);
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

    // The new token (window key 2047): CTA 15, the last warp of each head.
    const bool newer = rank == 15 && wk == 3;
    if (newer) {
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
#if !ASYNC
        if (g == 7) kr[3] = *(const uint4*)&newkv[h][0][16 * c];
        if (c == 3) vr[7] = *(const uint2*)&newkv[h][1][8 * g];
#endif
    }
#if ASYNC
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#pragma unroll
    for (int j = 0; j < 4; ++j) if (ASYNC & 1) kr[j] = skr[j][tid];
#pragma unroll
    for (int i = 0; i < 8; ++i) if (ASYNC & 2) vr[i] = svr[i][tid];
#endif
#if ASYNC
    if (newer) {
        if (g == 7) kr[3] = *(const uint4*)&newkv[h][0][16 * c];
        if (c == 3) vr[7] = *(const uint2*)&newkv[h][1][8 * g];
    }
#endif

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
        const unsigned owner = h2 * 8 + row;
        const unsigned rmb = mapa(mb, owner);
        sa4(mapa((unsigned)__cvta_generic_to_shared(&racc[rank][4 * dc]), owner), rmb, o[0], o[1], o[2], o[3]);
        if (dc == 0) sa2(mapa((unsigned)__cvta_generic_to_shared(&rml[rank][0]), owner), rmb, M, L);
    }

    // Merge query head `rank` from its 16 partials: one warp, two d per lane, no block barriers.
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
        *(__nv_bfloat162*)(out + (long)b * HQ * D + rank * D + 2 * lane) = __floats2bfloat162_rn(o0, o1);
    }
}
}  // namespace csla

void csl_decode(torch::Tensor qkv, torch::Tensor cs, torch::Tensor kc, torch::Tensor vc, torch::Tensor pt,
                torch::Tensor seq, torch::Tensor loc, torch::Tensor ks, torch::Tensor vs, torch::Tensor out, int64_t page,
                double qk_scale) {
    int batch = seq.size(0), mp = pt.size(1);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(16 * batch);
    cfg.blockDim = dim3(256);
    cfg.dynamicSmemBytes = ASYNC ? 32768 : 0;
    cfg.stream = c10::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 16; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.attrs = at; cfg.numAttrs = 1;
#define CSL_ARGS (const unsigned short*)qkv.data_ptr(), cs.data_ptr<float>(), (unsigned char*)kc.data_ptr(), \
    (unsigned char*)vc.data_ptr(), pt.data_ptr<int>(), seq.data_ptr<int>(), loc.data_ptr<long>(), ks.data_ptr<float>(), \
    vs.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(), mp, (float)qk_scale
    if (page == 1) {
        static bool set = false;
        if (!set) { cudaFuncSetAttribute(csla::decode<1>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1); cudaFuncSetAttribute(csla::decode<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768); set = true; }
        cudaLaunchKernelEx(&cfg, csla::decode<1>, CSL_ARGS);
    } else {
        static bool set = false;
        if (!set) { cudaFuncSetAttribute(csla::decode<64>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1); cudaFuncSetAttribute(csla::decode<64>, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768); set = true; }
        TORCH_CHECK(page == 64, "csl: page 1 or 64");
        cudaLaunchKernelEx(&cfg, csla::decode<64>, CSL_ARGS);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

int csl_clusters(int64_t minb) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(16 * 32); cfg.blockDim = dim3(256); cfg.dynamicSmemBytes = ASYNC ? 32768 : 0;
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 16; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.attrs = at; cfg.numAttrs = 1;
    cudaFuncSetAttribute(csla::decode<1>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaFuncSetAttribute(csla::decode<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 32768);
    int n = -1;
    cudaOccupancyMaxActiveClusters(&n, (void*)csla::decode<1>, &cfg);
    cudaFuncAttributes fa; cudaFuncGetAttributes(&fa, (void*)csla::decode<1>);
    printf("regs %d local %zu static smem %zu\n", fa.numRegs, fa.localSizeBytes, fa.sharedSizeBytes);
    return n;
}

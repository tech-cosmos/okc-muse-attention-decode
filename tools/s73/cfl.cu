// Batch-1 decode full attention as ONE kernel: NG clusters of 16 CTAs (8 warps each: head = warp / 4).
// The n = seq + 1 keys (the new token last) are cut into 128-key tiles, spread evenly over the 16 NG CTAs; a CTA
// gathers up to NS tiles at once with cp.async into shared memory (warp (h, wk) copies keys 32 wk + [0, 32) of
// head h, K rows 64 B, V rows 64 B with a bank swizzle), then runs QK^T / softmax / P@V per tile as it lands
// (mma.sync m16n8k32 E4M3, P in E4M3 with FA3's max offset 8, online rescale across tiles).
// Merge: the 4 warps of a head pre-merge in smem, each (head, row) partial is pushed with st.async + mbarrier
// complete_tx to its owner CTA (query head = cluster rank); the owner merges its 16 partials into the cluster's
// partial in global memory and bumps a per-head counter; the last cluster to arrive merges the NG cluster
// partials (RMSNorm, sigmoid gate, BF16) and resets the counter for the next launch.
// Fixed shapes: B 1, HQ 16, HKV 2, D 64, L 1, no RoPE.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#ifndef NS_
#define NS_ 3
#endif
#ifndef ABL
#define ABL 0
#endif
namespace cfl {
constexpr int HQ = 16, HKV = 2, D = 64, QKV_W = (2 * HQ + 2 * HKV) * D, NS = NS_;
constexpr int STAGE = 8 * 32 * D;  // bytes per stage per K or V

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
__device__ __forceinline__ void cpa16(unsigned dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
template <int N>
__device__ __forceinline__ void cpwait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }
__device__ __forceinline__ void cpwait_n(int n) {
    if (n <= 0) cpwait<0>();
    else if (n == 1) cpwait<1>();
    else if (n == 2) cpwait<2>();
    else cpwait<3>();
}
__device__ __forceinline__ float bf(unsigned short x) { return __uint_as_float((unsigned)x << 16); }
// V row r, 16-byte chunk q within a warp's 32 x 64 B block: rows r, r ^ 1 swap places when r & 2, chunks flip
// halves when r & 4, so the P@V reads (rows 2c + const, d 8g) of a half-warp hit 32 distinct banks.
__device__ __forceinline__ int voff(int r, int q) { return (r ^ ((r >> 1) & 1)) * D + 16 * (q ^ (((r >> 2) & 1) << 1)); }

template <int PAGE>
__global__ void __launch_bounds__(256) decode(const unsigned short* __restrict__ qkv, unsigned char* kc,
                                              unsigned char* vc, const int* __restrict__ pt,
                                              const int* __restrict__ seqs, const long* __restrict__ loc,
                                              const float* __restrict__ ks, const float* __restrict__ vs,
                                              __nv_bfloat16* __restrict__ out, float* gpart, int* cnt,
                                              float qk_scale) {
    extern __shared__ __align__(128) unsigned char smem[];  // [NS][K|V][8 warps][32][64]
    __shared__ __align__(16) float racc[16][D];
    __shared__ __align__(16) float rml[16][2];
    __shared__ __align__(16) float lml[2][4][8][2];
    __shared__ __align__(8) unsigned long long mbar;

    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
    const int h = warp >> 2, wk = warp & 3;
    unsigned rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    const int ng = gridDim.x / 16, cl = blockIdx.x / 16, cta = blockIdx.x;
    const unsigned mb = (unsigned)__cvta_generic_to_shared(&mbar);
    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n\tfence.mbarrier_init.release.cluster;" ::"r"(mb) : "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(mb), "r"(16 * (D * 4 + 8)) : "memory");
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

    // Page-table-independent loads first.
    const unsigned short* qp = qkv + (h * 8 + g) * D + 16 * c;
    const uint4 qa = *(const uint4*)qp, qb = *(const uint4*)(qp + 8);
    const unsigned short* kp = qkv + (HQ + h) * D + 2 * lane;
    const unsigned kw = *(const unsigned*)kp, vw = *(const unsigned*)(kp + HKV * D);
    const unsigned gw = *(const unsigned*)(qkv + (HQ + 2 * HKV) * D + rank * D + 2 * lane);
    const long nloc = loc[0];
    const float kscale = *ks, vscale = *vs;
    const int seq = seqs[0];
    const int n = seq + 1, ntile = (n + 127) >> 7, nct = 16 * ng;
    const int t0 = (int)((long)cta * ntile / nct), t1 = (int)((long)(cta + 1) * ntile / nct);
    const unsigned sbase = (unsigned)__cvta_generic_to_shared(smem);
    const unsigned kdst = sbase + warp * 32 * D, vdst = sbase + NS * STAGE + warp * 32 * D;

    // Issue tiles [r0, r0 + cnt) (stage s = tile - r0), one commit group per tile.
    auto issue = [&](int r0, int num) {
        for (int s = 0; s < num; ++s) {
            const int kb = ((r0 + s) << 7) + 32 * wk;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int r = 8 * j + g;
                const int t = min(kb + r, seq);
                const long slot = PAGE == 1 ? (long)pt[t] : (long)pt[t / PAGE] * PAGE + t % PAGE;
                const long src = slot * (HKV * D) + h * D + 16 * c;
                cpa16(kdst + s * STAGE + r * D + 16 * c, kc + src);
                cpa16(vdst + s * STAGE + voff(r, c), vc + src);
            }
            asm volatile("cp.async.commit_group;" ::: "memory");
        }
    };
    int r0 = t0, num = min(NS, t1 - t0);
    issue(r0, num);

#if ABL == 1
    {
        cpwait<0>();
        __syncwarp();
        unsigned z = qa.x ^ kw ^ vw ^ gw;
        for (int s = 0; s < num; ++s) z ^= *(const unsigned*)(smem + s * STAGE + warp * 32 * D + 4 * lane);
        asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
        if (z == 12345u) out[tid] = __float2bfloat16(1.f);
        return;
    }
#endif
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
#pragma unroll
    for (int w = 0; w < 4; ++w)
        q8[w] = pk8(x[4 * w + 1] * inv, x[4 * w] * inv) | (pk8(x[4 * w + 3] * inv, x[4 * w + 2] * inv) << 16);

    // The new token (key seq): the warp whose 32 keys hold it, in each head.
    const bool newer = (seq >> 7) >= t0 && (seq >> 7) < t1 && ((seq >> 5) & 3) == wk;
    unsigned short k8 = 0, v8 = 0;
    if (newer) {
        float k0 = bf(kw & 0xffff), k1 = bf(kw >> 16);
        float s = k0 * k0 + k1 * k1;
#pragma unroll
        for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
        float ki = rsqrtf(s / D + 1e-5f) / kscale;
        k8 = pk8(k1 * ki, k0 * ki);
        v8 = pk8(bf(vw >> 16) / vscale, bf(vw & 0xffff) / vscale);
        long slot = nloc * (HKV * D) + h * D + 2 * lane;
        *(unsigned short*)(kc + slot) = k8;
        *(unsigned short*)(vc + slot) = v8;
    }

    const float scale = qk_scale * kscale;
    float mo = -1e30f, l = 0.f;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
    for (int tile = t0; tile < t1; ++tile) {
        const int s = tile - r0;
        cpwait_n(num - 1 - s);
        const unsigned char* sk = smem + s * STAGE + warp * 32 * D;
        unsigned char* sv = smem + NS * STAGE + s * STAGE + warp * 32 * D;
        const int kb = (tile << 7) + 32 * wk;
        if (newer && kb == (seq & ~31)) {
            const int r = seq & 31;
            *(unsigned short*)(smem + s * STAGE + warp * 32 * D + r * D + 2 * lane) = k8;
            *(unsigned short*)(sv + voff(r, lane >> 3) + 2 * (lane & 7)) = v8;
        }
        __syncwarp();
        float sc[4][4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint4 kr = *(const uint4*)(sk + (8 * j + g) * D + 16 * c);
            sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.f;
            mma8(sc[j], q8[0], 0u, q8[1], 0u, kr.x, kr.y);
            mma8(sc[j], q8[2], 0u, q8[3], 0u, kr.z, kr.w);
        }
        float m = mo + 8.f;
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                sc[j][e] = kb + 8 * j + 2 * c + e < n ? sc[j][e] * scale : -INFINITY;
                m = fmaxf(m, sc[j][e]);
            }
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1));
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 2));
        const float mn = m - 8.f;  // FA3's FP8 max offset: P <= 2^8
        const float alpha = exp2f(mo - mn);
        mo = mn;
        l *= alpha;
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int e = 0; e < 2; ++e) { sc[j][e] = exp2f(sc[j][e] - mn); l += sc[j][e]; }
        const float a0 = __shfl_sync(0xffffffff, alpha, 8 * c), a1 = __shfl_sync(0xffffffff, alpha, 8 * c + 4);
        const unsigned pb0 = pk8(sc[0][1], sc[0][0]) | (pk8(sc[1][1], sc[1][0]) << 16);
        const unsigned pb1 = pk8(sc[2][1], sc[2][0]) | (pk8(sc[3][1], sc[3][0]) << 16);
        uint2 vr[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int r = 8 * (i >> 1) + 2 * c + (i & 1);
            vr[i] = *(const uint2*)(sv + voff(r, g >> 1) + 8 * (g & 1));
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            acc[i][0] *= a0; acc[i][2] *= a0; acc[i][1] *= a1; acc[i][3] *= a1;
            const int bsel = (i & 1) ? 0x7362 : 0x5140;
            unsigned w[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) w[k] = (i < 2) ? vr[k].x : vr[k].y;
            unsigned t01 = __byte_perm(w[0], w[1], bsel), t23 = __byte_perm(w[2], w[3], bsel);
            unsigned t45 = __byte_perm(w[4], w[5], bsel), t67 = __byte_perm(w[6], w[7], bsel);
            mma8(acc[i], __byte_perm(t01, t23, 0x5410), __byte_perm(t01, t23, 0x7632), __byte_perm(t45, t67, 0x5410),
                 __byte_perm(t45, t67, 0x7632), pb0, pb1);
        }
        if (s == num - 1 && tile + 1 < t1) {  // refill all stages with the next round
            __syncwarp();
            r0 = tile + 1;
            num = min(NS, t1 - r0);
            issue(r0, num);
        }
    }
    l += __shfl_xor_sync(0xffffffff, l, 1);
    l += __shfl_xor_sync(0xffffffff, l, 2);
#if ABL == 2
    {
        asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
        float z = acc[0][0] + acc[1][1] + acc[2][2] + acc[3][3] + l;
        if (z == 12345.f) out[tid] = __float2bfloat16(1.f);
        return;
    }
#endif

    // Pre-merge the 4 warps of each head (warp w's 8 x 64 floats reuse its own stage-0 K block).
    float (*lpart)[4][8][D] = reinterpret_cast<float (*)[4][8][D]>(smem);
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
        if (ng > 1) {
            // Cluster partial -> global; the last cluster for this query head merges all of them.
            float* gp = gpart + ((long)rank * ng + cl) * (D + 2);
            *(float2*)(gp + 2 + 2 * lane) = make_float2(o0, o1);
            if (lane == 0) *(float2*)gp = make_float2(M, den);
            __threadfence();
            __syncwarp();
            int old = 0;
            if (lane == 0) old = atomicAdd(cnt + rank, 1);
            old = __shfl_sync(0xffffffff, old, 0);
            if (old != ng - 1) return;
            __threadfence();
            if (lane == 0) cnt[rank] = 0;
            const float* gq = gpart + (long)rank * ng * (D + 2);
            M = -1e30f;
            for (int s = 0; s < ng; ++s) M = fmaxf(M, __ldcg(gq + s * (D + 2)));
            den = 0.f; o0 = 0.f; o1 = 0.f;
            for (int s = 0; s < ng; ++s) {
                const float* p = gq + s * (D + 2);
                float w = exp2f(__ldcg(p) - M);
                den += w * __ldcg(p + 1);
                float2 a = __ldcg((const float2*)(p + 2 + 2 * lane));
                o0 += w * a.x; o1 += w * a.y;
            }
        }
        float r = vscale / den;
        o0 *= r; o1 *= r;
        float s2 = o0 * o0 + o1 * o1;
#pragma unroll
        for (int o = 16; o; o >>= 1) s2 += __shfl_xor_sync(0xffffffff, s2, o);
        float ni = rsqrtf(s2 / D + 1e-5f);
        const float g0 = bf(gw & 0xffff), g1 = bf(gw >> 16);
        o0 *= ni / (1.f + __expf(-g0));
        o1 *= ni / (1.f + __expf(-g1));
        *(__nv_bfloat162*)(out + rank * D + 2 * lane) = __floats2bfloat162_rn(o0, o1);
    }
}
}  // namespace cfl

void cfl_decode(torch::Tensor qkv, torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seq,
                torch::Tensor loc, torch::Tensor ks, torch::Tensor vs, torch::Tensor out, torch::Tensor gpart,
                torch::Tensor cnt, int64_t page, double qk_scale, int64_t ng) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(16 * ng);
    cfg.blockDim = dim3(256);
    cfg.dynamicSmemBytes = 2 * cfl::NS * cfl::STAGE;
    cfg.stream = c10::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 16; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.attrs = at; cfg.numAttrs = 1;
    auto go = [&](auto kern) {
        cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)cfg.dynamicSmemBytes);
        cudaLaunchKernelEx(&cfg, kern, (const unsigned short*)qkv.data_ptr(), (unsigned char*)kc.data_ptr(),
                           (unsigned char*)vc.data_ptr(), pt.data_ptr<int>(), seq.data_ptr<int>(), loc.data_ptr<long>(),
                           ks.data_ptr<float>(), vs.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(),
                           gpart.data_ptr<float>(), cnt.data_ptr<int>(), (float)qk_scale);
    };
    TORCH_CHECK(page == 1 || page == 64, "cfl: page 1 or 64");
    if (page == 1) go(cfl::decode<1>); else go(cfl::decode<64>);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

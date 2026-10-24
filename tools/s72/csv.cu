// Verify-block (L 8) sliding attention as ONE kernel: one 16-CTA cluster per (request, KV head).
// Rows of a head: r = token * 8 + query head (64). CTA rank r covers window slots [128 r, 128 r + 128) (slot w = key
// seq - 2047 + w; slot 2047 is the block's first token, handled with the new tokens). 8 warps: rq = warp & 3 is the
// m-tile (tokens 2rq, 2rq + 1, all 8 heads), kh = warp >> 2 the 64-key half (2 chunks of 32). CTA 15's kh 1 warps
// also run the block's 8 new tokens (causal) as a third chunk. One joint max per warp over its chunks (no rescale).
// QK^T and P@V are mma.sync m16n8k32 E4M3 as in csl.cu. Partials: the 2 key halves pre-merge in smem, then
// st.async (mbarrier complete_tx) to the owner CTA (rows 4 o .. 4 o + 3), which merges 16 sources per row.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>

namespace csv {
constexpr int HQ = 16, HKV = 2, D = 64, W = 2048, L = 8, QKV_W = (2 * HQ + 2 * HKV) * D;

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
__device__ __forceinline__ float bf(unsigned short x) { return __uint_as_float((unsigned)x << 16); }

struct Smem {
    float lpart[2][64][D];  // pre-merge: key half, row, d
    float lml[2][64][2];
    float racc[4][16][D];   // owner: owned row, source rank, d
    float rml[4][16][2];
    unsigned char newk[L][D], newv[L][D];
    unsigned long long mbar;
};

// S for one 32-key chunk: sc[j][0..1] token 2rq (keys 8j + 2c + e), sc[j][2..3] token 2rq + 1.
__device__ __forceinline__ void qk(float (*sc)[4], const unsigned* qf, const uint4* kr) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.f;
        mma8(sc[j], qf[0], qf[1], qf[2], qf[3], kr[j].x, kr[j].y);
        mma8(sc[j], qf[4], qf[5], qf[6], qf[7], kr[j].z, kr[j].w);
    }
}
// acc[s][i] += V^T(chunk) P^T(token s of the m-tile).
__device__ __forceinline__ void pv(float (*acc)[4][4], float (*sc)[4], const uint2* vr) {
    unsigned a[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int bsel = (i & 1) ? 0x7362 : 0x5140;
        unsigned w[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) w[k] = (i < 2) ? vr[k].x : vr[k].y;
        unsigned t01 = __byte_perm(w[0], w[1], bsel), t23 = __byte_perm(w[2], w[3], bsel);
        unsigned t45 = __byte_perm(w[4], w[5], bsel), t67 = __byte_perm(w[6], w[7], bsel);
        a[i][0] = __byte_perm(t01, t23, 0x5410); a[i][1] = __byte_perm(t01, t23, 0x7632);
        a[i][2] = __byte_perm(t45, t67, 0x5410); a[i][3] = __byte_perm(t45, t67, 0x7632);
    }
#pragma unroll
    for (int s = 0; s < 2; ++s) {
        const unsigned pb0 = pk8(sc[0][2 * s + 1], sc[0][2 * s]) | (pk8(sc[1][2 * s + 1], sc[1][2 * s]) << 16);
        const unsigned pb1 = pk8(sc[2][2 * s + 1], sc[2][2 * s]) | (pk8(sc[3][2 * s + 1], sc[3][2 * s]) << 16);
#pragma unroll
        for (int i = 0; i < 4; ++i) mma8(acc[s][i], a[i][0], a[i][1], a[i][2], a[i][3], pb0, pb1);
    }
}

template <int PAGE>
__global__ void __launch_bounds__(256) verify(const unsigned short* __restrict__ qkv, const float* __restrict__ cs,
                                              unsigned char* kc, unsigned char* vc, const int* __restrict__ pt,
                                              const int* __restrict__ seqs, const long* __restrict__ loc,
                                              const float* __restrict__ ks, const float* __restrict__ vs,
                                              __nv_bfloat16* __restrict__ out, int MP, float qk_scale) {
    extern __shared__ __align__(16) unsigned char smraw[];
    Smem& sm = *reinterpret_cast<Smem*>(smraw);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
    const int rq = warp & 3, kh = warp >> 2;
    unsigned rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    const int b = blockIdx.x / 32, h = (blockIdx.x / 16) & 1;
    const unsigned mb = (unsigned)__cvta_generic_to_shared(&sm.mbar);
    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n\tfence.mbarrier_init.release.cluster;" ::"r"(mb) : "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(mb), "r"(4 * 16 * (4 * 16 * 4 + 8))
                     : "memory");
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

    // Page-table-independent loads first: queries (tokens 2rq + s, head g, d 16c..16c+15), new tokens, scales.
    const long tok0 = (long)b * L + 2 * rq;
    uint4 qr[2][2];
#pragma unroll
    for (int s = 0; s < 2; ++s) {
        const unsigned short* qp = qkv + (tok0 + s) * QKV_W + (h * 8 + g) * D + 16 * c;
        qr[s][0] = *(const uint4*)qp; qr[s][1] = *(const uint4*)(qp + 8);
    }
    const bool newer = rank == 15 && kh == 1;  // these 4 warps make tokens 2rq, 2rq + 1 and run the new chunk
    unsigned nkw[2], nvw[2];
#pragma unroll
    for (int s = 0; s < 2; ++s) {
        const unsigned short* kp = qkv + (tok0 + s) * QKV_W + (HQ + h) * D + 2 * lane;
        nkw[s] = *(const unsigned*)kp; nvw[s] = *(const unsigned*)(kp + HKV * D);
    }
    const float kscale = *ks, vscale = *vs;
    const int seq = seqs[b];
    float4 cq[2][4];
#pragma unroll
    for (int s = 0; s < 2; ++s) {
        const float* p = cs + (long)(seq + 2 * rq + s) * D + 8 * c;
        cq[s][0] = *(const float4*)p; cq[s][1] = *(const float4*)(p + 4);
        cq[s][2] = *(const float4*)(p + 32); cq[s][3] = *(const float4*)(p + 36);
    }
    float nco[2], nsi[2];
#pragma unroll
    for (int s = 0; s < 2; ++s) {
        nco[s] = cs[(long)(seq + 2 * rq + s) * D + lane]; nsi[s] = cs[(long)(seq + 2 * rq + s) * D + 32 + lane];
    }
    long nloc[2] = {loc[tok0], loc[tok0 + 1]};
    const int base = seq - (W - 1);
    const int wb = rank * 128 + kh * 64;
    const int* ptr = pt + (long)b * MP;
    const int tmax = max(seq - 1, 0);
    uint4 kr[2][4];
    uint2 vr[2][8];
    {
        long kslot[2][4], vslot[2][8];
#pragma unroll
        for (int ch = 0; ch < 2; ++ch) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                int t = min(max(base + wb + 32 * ch + 8 * j + g, 0), tmax);
                kslot[ch][j] = PAGE == 1 ? (long)ptr[t] : (long)ptr[t / PAGE] * PAGE + t % PAGE;
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                int t = min(max(base + wb + 32 * ch + 8 * (i >> 1) + 2 * c + (i & 1), 0), tmax);
                vslot[ch][i] = PAGE == 1 ? (long)ptr[t] : (long)ptr[t / PAGE] * PAGE + t % PAGE;
            }
        }
#pragma unroll
        for (int ch = 0; ch < 2; ++ch) {
#pragma unroll
            for (int j = 0; j < 4; ++j) kr[ch][j] = *(const uint4*)(kc + kslot[ch][j] * (HKV * D) + h * D + 16 * c);
#pragma unroll
            for (int i = 0; i < 8; ++i) vr[ch][i] = *(const uint2*)(vc + vslot[ch][i] * (HKV * D) + h * D + 8 * g);
        }
    }

    // Queries: rmsnorm, RoPE at seq + token, E4M3. A fragment: qf[4 ks + {0: tok s0 lo, 1: tok s1 lo, 2: s0 hi, 3: s1 hi}].
    unsigned qf[8];
#pragma unroll
    for (int s = 0; s < 2; ++s) {
        float x[16];
        unsigned u[8] = {qr[s][0].x, qr[s][0].y, qr[s][0].z, qr[s][0].w, qr[s][1].x, qr[s][1].y, qr[s][1].z, qr[s][1].w};
#pragma unroll
        for (int i = 0; i < 8; ++i) { x[2 * i] = bf(u[i] & 0xffff); x[2 * i + 1] = bf(u[i] >> 16); }
        float ss = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) ss += x[i] * x[i];
        ss += __shfl_xor_sync(0xffffffff, ss, 1);
        ss += __shfl_xor_sync(0xffffffff, ss, 2);
        const float inv = rsqrtf(ss / D + 1e-5f);
        float cc[8] = {cq[s][0].x, cq[s][0].y, cq[s][0].z, cq[s][0].w, cq[s][1].x, cq[s][1].y, cq[s][1].z, cq[s][1].w};
        float sn[8] = {cq[s][2].x, cq[s][2].y, cq[s][2].z, cq[s][2].w, cq[s][3].x, cq[s][3].y, cq[s][3].z, cq[s][3].w};
        float y[16];
#pragma unroll
        for (int p = 0; p < 8; ++p) {
            y[2 * p] = (x[2 * p] * cc[p] - x[2 * p + 1] * sn[p]) * inv;
            y[2 * p + 1] = (x[2 * p + 1] * cc[p] + x[2 * p] * sn[p]) * inv;
        }
        unsigned q8[4];
#pragma unroll
        for (int w = 0; w < 4; ++w) q8[w] = pk8(y[4 * w + 1], y[4 * w]) | (pk8(y[4 * w + 3], y[4 * w + 2]) << 16);
        qf[s] = q8[0]; qf[2 + s] = q8[1]; qf[4 + s] = q8[2]; qf[6 + s] = q8[3];
    }

    const float scale = qk_scale * kscale;
    float sc[3][4][4];
    qk(sc[0], qf, kr[0]);
    qk(sc[1], qf, kr[1]);
    uint4 krn[4];
    uint2 vrn[8];
    if (newer) {
        // New tokens 2rq + s of head h: rmsnorm, RoPE, E4M3; to the cache and to smem for the new chunk.
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            float k0 = bf(nkw[s] & 0xffff), k1 = bf(nkw[s] >> 16);
            float q = k0 * k0 + k1 * k1;
#pragma unroll
            for (int o = 16; o; o >>= 1) q += __shfl_xor_sync(0xffffffff, q, o);
            float ki = rsqrtf(q / D + 1e-5f) / kscale;
            unsigned short k8 = pk8((k1 * nco[s] + k0 * nsi[s]) * ki, (k0 * nco[s] - k1 * nsi[s]) * ki);
            unsigned short v8 = pk8(bf(nvw[s] >> 16) / vscale, bf(nvw[s] & 0xffff) / vscale);
            *(unsigned short*)&sm.newk[2 * rq + s][2 * lane] = k8;
            *(unsigned short*)&sm.newv[2 * rq + s][2 * lane] = v8;
            long slot = nloc[s] * (HKV * D) + h * D + 2 * lane;
            *(unsigned short*)(kc + slot) = k8;
            *(unsigned short*)(vc + slot) = v8;
        }
        asm volatile("bar.sync 1, 128;" ::: "memory");
        krn[0] = *(const uint4*)&sm.newk[g][16 * c];
        krn[1] = krn[2] = krn[3] = make_uint4(0, 0, 0, 0);
        vrn[0] = *(const uint2*)&sm.newv[2 * c][8 * g];
        vrn[1] = *(const uint2*)&sm.newv[2 * c + 1][8 * g];
#pragma unroll
        for (int i = 2; i < 8; ++i) vrn[i] = make_uint2(0, 0);
        qk(sc[2], qf, krn);
    }

    // Masks, joint max over the warp's chunks, P (E4M3 with FA3's offset 8), row sums.
    float m[2] = {-1e30f, -1e30f};
#pragma unroll
    for (int ch = 0; ch < 3; ++ch) {
        if (ch == 2 && !newer) break;
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int tok = 2 * rq + (e >> 1), k = 8 * j + 2 * c + (e & 1);
                bool ok;
                if (ch < 2) {
                    const int slot = wb + 32 * ch + k;
                    ok = slot >= tok && slot < W - 1 && base + slot >= 0;
                } else {
                    ok = k <= tok;
                }
                sc[ch][j][e] = ok ? sc[ch][j][e] * scale : -INFINITY;
                m[e >> 1] = fmaxf(m[e >> 1], sc[ch][j][e]);
            }
    }
    float mo[2], l[2] = {0.f, 0.f};
#pragma unroll
    for (int s = 0; s < 2; ++s) {
        m[s] = fmaxf(m[s], __shfl_xor_sync(0xffffffff, m[s], 1));
        m[s] = fmaxf(m[s], __shfl_xor_sync(0xffffffff, m[s], 2));
        mo[s] = m[s] - 8.f;
    }
#pragma unroll
    for (int ch = 0; ch < 3; ++ch) {
        if (ch == 2 && !newer) break;
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) { sc[ch][j][e] = exp2f(sc[ch][j][e] - mo[e >> 1]); l[e >> 1] += sc[ch][j][e]; }
    }
#pragma unroll
    for (int s = 0; s < 2; ++s) {
        l[s] += __shfl_xor_sync(0xffffffff, l[s], 1);
        l[s] += __shfl_xor_sync(0xffffffff, l[s], 2);
    }
    float acc[2][4][4];
#pragma unroll
    for (int s = 0; s < 2; ++s)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[s][i][0] = acc[s][i][1] = acc[s][i][2] = acc[s][i][3] = 0.f;
    pv(acc, sc[0], vr[0]);
    pv(acc, sc[1], vr[1]);
    if (newer) pv(acc, sc[2], vrn);

    // Pre-merge the two key halves: rows (token 2rq + s) * 8 + head; acc[s][i][e] is head 2c + (e & 1), d 8g + 2i + (e >> 1).
#pragma unroll
    for (int s = 0; s < 2; ++s)
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            float* r = &sm.lpart[kh][(2 * rq + s) * 8 + 2 * c + e][8 * g];
            *(float4*)r = make_float4(acc[s][0][e], acc[s][0][e + 2], acc[s][1][e], acc[s][1][e + 2]);
            *(float4*)(r + 4) = make_float4(acc[s][2][e], acc[s][2][e + 2], acc[s][3][e], acc[s][3][e + 2]);
        }
    if (c == 0) {
#pragma unroll
        for (int s = 0; s < 2; ++s) *(float2*)&sm.lml[kh][(2 * rq + s) * 8 + g][0] = make_float2(mo[s], l[s]);
    }
    __syncthreads();
    {
        const int row = tid >> 2, dq = tid & 3;
        const float2 a = *(const float2*)&sm.lml[0][row][0], bb = *(const float2*)&sm.lml[1][row][0];
        const float M = fmaxf(a.x, bb.x), w0 = exp2f(a.x - M), w1 = exp2f(bb.x - M), Lr = w0 * a.y + w1 * bb.y;
        asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
        const unsigned owner = row >> 2, orow = row & 3, rmb = mapa(mb, owner);
        const unsigned dst = mapa((unsigned)__cvta_generic_to_shared(&sm.racc[orow][rank][16 * dq]), owner);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const float4 x = *(const float4*)&sm.lpart[0][row][16 * dq + 4 * q];
            const float4 y = *(const float4*)&sm.lpart[1][row][16 * dq + 4 * q];
            sa4(dst + 16 * q, rmb, w0 * x.x + w1 * y.x, w0 * x.y + w1 * y.y, w0 * x.z + w1 * y.z, w0 * x.w + w1 * y.w);
        }
        if (dq == 0) sa2(mapa((unsigned)__cvta_generic_to_shared(&sm.rml[orow][rank][0]), owner), rmb, M, Lr);
    }

    // Owner: rows 4 rank + warp (warps 0-3), 16 sources each.
    if (warp < 4) {
        const int row = 4 * rank + warp, tok = row >> 3, qh = h * 8 + (row & 7);
        const unsigned gw = *(const unsigned*)(qkv + ((long)b * L + tok) * QKV_W + (HQ + 2 * HKV) * D + qh * D + 2 * lane);
        asm volatile("{\n\t.reg .pred p;\n\tW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n\t@!p bra W;\n\t}"
                     ::"r"(mb) : "memory");
        float M = -1e30f;
#pragma unroll
        for (int s = 0; s < 16; ++s) M = fmaxf(M, sm.rml[warp][s][0]);
        float den = 0.f, o0 = 0.f, o1 = 0.f;
#pragma unroll
        for (int s = 0; s < 16; ++s) {
            float w = exp2f(sm.rml[warp][s][0] - M);
            den += w * sm.rml[warp][s][1];
            float2 a = *(const float2*)&sm.racc[warp][s][2 * lane];
            o0 += w * a.x; o1 += w * a.y;
        }
        float r = vscale / den;
        o0 *= r; o1 *= r;
        float s2 = o0 * o0 + o1 * o1;
#pragma unroll
        for (int o = 16; o; o >>= 1) s2 += __shfl_xor_sync(0xffffffff, s2, o);
        float ni = rsqrtf(s2 / D + 1e-5f);
        o0 *= ni / (1.f + __expf(-bf(gw & 0xffff)));
        o1 *= ni / (1.f + __expf(-bf(gw >> 16)));
        *(__nv_bfloat162*)(out + ((long)b * L + tok) * HQ * D + qh * D + 2 * lane) = __floats2bfloat162_rn(o0, o1);
    }
}
}  // namespace csv

void csv_verify(torch::Tensor qkv, torch::Tensor cs, torch::Tensor kc, torch::Tensor vc, torch::Tensor pt,
                torch::Tensor seq, torch::Tensor loc, torch::Tensor ks, torch::Tensor vs, torch::Tensor out, int64_t page,
                double qk_scale) {
    const int batch = seq.size(0);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(32 * batch);
    cfg.blockDim = dim3(256);
    cfg.dynamicSmemBytes = sizeof(csv::Smem);
    cfg.stream = c10::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 16; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.attrs = at; cfg.numAttrs = 1;
    TORCH_CHECK(page == 1 || page == 64, "csv: page 1 or 64");
    auto f = page == 1 ? csv::verify<1> : csv::verify<64>;
    static bool set[2] = {false, false};
    if (!set[page == 1]) {
        cudaFuncSetAttribute(f, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
        cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(csv::Smem));
        set[page == 1] = true;
    }
    cudaLaunchKernelEx(&cfg, f, (const unsigned short*)qkv.data_ptr(), cs.data_ptr<float>(), (unsigned char*)kc.data_ptr(),
                       (unsigned char*)vc.data_ptr(), pt.data_ptr<int>(), seq.data_ptr<int>(), loc.data_ptr<long>(),
                       ks.data_ptr<float>(), vs.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(), (int)pt.size(1),
                       (float)qk_scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

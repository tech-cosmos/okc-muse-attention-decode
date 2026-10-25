// agent2 s7 dnm: decode sliding attention with NO key split: one CTA per request (both KV heads), NW warps, warp w owns
// window keys [W/NW w, W/NW (w + 1)) and streams them in 32-key stages (K and V 128-B rows of both heads, 8 KB) through
// a private NS-stage cp.async ring; csl's math per stage and head (mma.sync m16n8k32 E4M3, P in E4M3 with FA3's max
// offset 8, V^T by byte permutes) with an online softmax (FP32 m, l, acc), then the NW warps' partials merge in shared
// memory (RMSNorm, sigmoid gate, BF16). No merge kernel, no clusters. Fixed shapes: HQ 16, HKV 2, D 64, window 2048, L 1.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#ifndef DNM_NOGATH
#define DNM_NOGATH 0
#endif
namespace dnm {
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
__device__ __forceinline__ float bf(unsigned short x) { return __uint_as_float((unsigned)x << 16); }
__device__ __forceinline__ void cp16(unsigned dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
// 16-B chunk swizzle of a stage row (key): K reads (keys 8j + g, 8 lanes per phase) and V reads (keys 2c + e) are
// conflict-free.
__device__ __forceinline__ int swz(int key) { return ((key & 1) << 2) ^ (((key >> 1) & 3) << 1); }

template <int PAGE, int NW, int NS>
__global__ void __launch_bounds__(NW * 32, 1) decode(const unsigned short* __restrict__ qkv, const float* __restrict__ cs,
                                                     unsigned char* kc, unsigned char* vc, const int* __restrict__ pt,
                                                     const int* __restrict__ seqs, const long* __restrict__ loc,
                                                     const float* __restrict__ ks, const float* __restrict__ vs,
                                                     __nv_bfloat16* __restrict__ out, int MP, float qk_scale) {
    constexpr int KPW = W / NW, NST = KPW / 32, STB = 8192;
    extern __shared__ __align__(128) unsigned char smem[];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
    const int b = blockIdx.x;
    const int seq = seqs[b];
    const int base = seq - (W - 1), tmax = max(seq - 1, 0), w0 = warp * KPW;
    const int* ptr = pt + (long)b * MP;
    // Page-table entries of the warp's keys go to shared memory (after the ring + merge area), one per key.
    int* sl = (int*)(smem + NW * NS * STB) + warp * KPW;
#pragma unroll
    for (int s = 0; s < NST; ++s) {
        int t = min(max(base + w0 + 32 * s + lane, 0), tmax);
        sl[32 * s + lane] = PAGE == 1 ? ptr[t] : ptr[t / PAGE] * PAGE + t % PAGE;
    }
    __syncwarp();
    const unsigned ring = (unsigned)__cvta_generic_to_shared(smem) + warp * NS * STB;
    auto issue = [&](int s) {
        const unsigned st = ring + (s % NS) * STB;
#pragma unroll
        for (int n = 0; n < 8; ++n) {
            const int key = 4 * n + (lane >> 3), ch = lane & 7;
            const long slot = sl[32 * s + key];
            const unsigned o = key * 128 + ((ch ^ swz(key)) << 4);
            if (!DNM_NOGATH) cp16(st + o, kc + slot * (HKV * D) + 16 * ch);
            if (!DNM_NOGATH) cp16(st + 4096 + o, vc + slot * (HKV * D) + 16 * ch);
        }
    };
#pragma unroll
    for (int s = 0; s < NS - 1; ++s) {
        if (s < NST) issue(s);
        asm volatile("cp.async.commit_group;" ::: "memory");
    }

    // Queries of both heads: thread (g, c) holds row h 8 + g, d 16 c .. 16 c + 15 (QK-norm, RoPE, E4M3), as csl.
    const float* csr = cs + (long)seq * D + 8 * c;
    const float4 c0 = *(const float4*)csr, c1 = *(const float4*)(csr + 4);
    const float4 s0 = *(const float4*)(csr + 32), s1 = *(const float4*)(csr + 36);
    unsigned q8[2][4];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const unsigned short* qp = qkv + (long)b * QKV_W + (h * 8 + g) * D + 16 * c;
        const uint4 qa = *(const uint4*)qp, qb = *(const uint4*)(qp + 8);
        float x[16];
        unsigned u[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
#pragma unroll
        for (int i = 0; i < 8; ++i) { x[2 * i] = bf(u[i] & 0xffff); x[2 * i + 1] = bf(u[i] >> 16); }
        float ss = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) ss += x[i] * x[i];
        ss += __shfl_xor_sync(0xffffffff, ss, 1);
        ss += __shfl_xor_sync(0xffffffff, ss, 2);
        const float inv = rsqrtf(ss / D + 1e-5f);
        float cc[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
        float sn[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
        float y[16];
#pragma unroll
        for (int p = 0; p < 8; ++p) {
            y[2 * p] = (x[2 * p] * cc[p] - x[2 * p + 1] * sn[p]) * inv;
            y[2 * p + 1] = (x[2 * p + 1] * cc[p] + x[2 * p] * sn[p]) * inv;
        }
#pragma unroll
        for (int w = 0; w < 4; ++w) q8[h][w] = pk8(y[4 * w + 1], y[4 * w]) | (pk8(y[4 * w + 3], y[4 * w + 2]) << 16);
    }
    const float kscale = *ks, vscale = *vs;
    // The new token (window key 2047 = the last warp's last stage, row 31): lane covers bytes 4 lane .. 4 lane + 3
    // of the 128-B row (head lane / 16, d 4 (lane % 16) ..).
    unsigned k8 = 0, v8 = 0;
    if (warp == NW - 1) {
        const int h = lane >> 4, j = lane & 15;
        const unsigned short* kp = qkv + (long)b * QKV_W + (HQ + h) * D + 4 * j;
        const uint2 kw = *(const uint2*)kp, vw = *(const uint2*)(kp + HKV * D);
        float k[4] = {bf(kw.x & 0xffff), bf(kw.x >> 16), bf(kw.y & 0xffff), bf(kw.y >> 16)};
        float s = k[0] * k[0] + k[1] * k[1] + k[2] * k[2] + k[3] * k[3];
#pragma unroll
        for (int o = 8; o; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
        const float ki = rsqrtf(s / D + 1e-5f) / kscale;
        const float2 co = *(const float2*)(cs + (long)seq * D + 2 * j), si = *(const float2*)(cs + (long)seq * D + 32 + 2 * j);
        float r0 = (k[0] * co.x - k[1] * si.x) * ki, r1 = (k[1] * co.x + k[0] * si.x) * ki;
        float r2 = (k[2] * co.y - k[3] * si.y) * ki, r3 = (k[3] * co.y + k[2] * si.y) * ki;
        k8 = pk8(r1, r0) | (pk8(r3, r2) << 16);
        v8 = pk8(bf(vw.x >> 16) / vscale, bf(vw.x & 0xffff) / vscale) |
             (pk8(bf(vw.y >> 16) / vscale, bf(vw.y & 0xffff) / vscale) << 16);
        const long sl = loc[b] * (HKV * D) + 4 * lane;
        *(unsigned*)(kc + sl) = k8;
        *(unsigned*)(vc + sl) = v8;
    }

    const float scale = qk_scale * kscale;
    float mrun[2] = {-1e30f, -1e30f}, lrun[2] = {0.f, 0.f};
    float acc[2][4][4];
#pragma unroll
    for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[h][i][0] = acc[h][i][1] = acc[h][i][2] = acc[h][i][3] = 0.f;

#pragma unroll 1
    for (int s = 0; s < NST; ++s) {
        if (s + NS - 1 < NST) issue(s + NS - 1);
        asm volatile("cp.async.commit_group;" ::: "memory");
        asm volatile("cp.async.wait_group %0;" ::"n"(NS - 1) : "memory");
        __syncwarp();
        const unsigned char* K = smem + warp * NS * STB + (s % NS) * STB;
        const unsigned char* V = K + 4096;
        if (s == NST - 1 && warp == NW - 1) {
            const int o = 31 * 128 + ((((lane >> 2) ^ swz(31)) << 4) | (4 * (lane & 3)));
            *(unsigned*)(K + o) = k8;
            *(unsigned*)(V + o) = v8;
            __syncwarp();
        }
#ifndef DNM_NOCOMP
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            uint4 kr[4];
            uint2 vr[8];
#pragma unroll
            for (int j = 0; j < 4; ++j) kr[j] = *(const uint4*)(K + (8 * j + g) * 128 + (((4 * h + c) ^ swz(g)) << 4));
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int key = 8 * (i >> 1) + 2 * c + (i & 1);
                vr[i] = *(const uint2*)(V + key * 128 + (((4 * h + (g >> 1)) ^ swz(key)) << 4) + 8 * (g & 1));
            }
            float sc[4][4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                sc[j][0] = sc[j][1] = sc[j][2] = sc[j][3] = 0.f;
                mma8(sc[j], q8[h][0], 0u, q8[h][1], 0u, kr[j].x, kr[j].y);
                mma8(sc[j], q8[h][2], 0u, q8[h][3], 0u, kr[j].z, kr[j].w);
            }
            float m = mrun[h];
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const bool ok = base + w0 + 32 * s + 8 * j + 2 * c + e >= 0;
                    sc[j][e] = ok ? sc[j][e] * scale : -INFINITY;
                    m = fmaxf(m, sc[j][e]);
                }
            m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1));
            m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 2));
            const float alpha = exp2f(mrun[h] - m);
            mrun[h] = m;
            const float mo = m - 8.f;  // FA3's FP8 max offset: P <= 2^8
            float l = 0.f;
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int e = 0; e < 2; ++e) { sc[j][e] = exp2f(sc[j][e] - mo); l += sc[j][e]; }
            lrun[h] = lrun[h] * alpha + l;
            const unsigned pb0 = pk8(sc[0][1], sc[0][0]) | (pk8(sc[1][1], sc[1][0]) << 16);
            const unsigned pb1 = pk8(sc[2][1], sc[2][0]) | (pk8(sc[3][1], sc[3][0]) << 16);
            const float a0 = __shfl_sync(0xffffffff, alpha, 8 * c), a1 = __shfl_sync(0xffffffff, alpha, 8 * c + 4);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc[h][i][0] *= a0; acc[h][i][1] *= a1; acc[h][i][2] *= a0; acc[h][i][3] *= a1;
                const int bsel = (i & 1) ? 0x7362 : 0x5140;
                unsigned w[8];
#pragma unroll
                for (int k = 0; k < 8; ++k) w[k] = (i < 2) ? vr[k].x : vr[k].y;
                unsigned t01 = __byte_perm(w[0], w[1], bsel), t23 = __byte_perm(w[2], w[3], bsel);
                unsigned t45 = __byte_perm(w[4], w[5], bsel), t67 = __byte_perm(w[6], w[7], bsel);
                mma8(acc[h][i], __byte_perm(t01, t23, 0x5410), __byte_perm(t01, t23, 0x7632),
                     __byte_perm(t45, t67, 0x5410), __byte_perm(t45, t67, 0x7632), pb0, pb1);
            }
        }
#else
        acc[0][0][0] += K[lane * 4] + V[lane * 4];
#endif
        __syncwarp();
    }
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();

    // Merge the NW warps per (head, row): partials [NW][2][8][D] and (m - 8, l) [NW][2][8] in the ring's smem.
    float* part = (float*)smem;
    float* pml = part + NW * 2 * 8 * D;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        float lsum = lrun[h];
        lsum += __shfl_xor_sync(0xffffffff, lsum, 1);
        lsum += __shfl_xor_sync(0xffffffff, lsum, 2);
        float* lp = part + ((warp * 2 + h) * 8) * D;
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            float* r = lp + (2 * c + e) * D + 8 * g;
            *(float4*)r = make_float4(acc[h][0][e], acc[h][0][e + 2], acc[h][1][e], acc[h][1][e + 2]);
            *(float4*)(r + 4) = make_float4(acc[h][2][e], acc[h][2][e + 2], acc[h][3][e], acc[h][3][e + 2]);
        }
        if (c == 0) *(float2*)&pml[((warp * 2 + h) * 8 + g) * 2] = make_float2(mrun[h], lsum);
    }
    __syncthreads();
    for (int it = tid; it < 2 * 8 * 16; it += NW * 32) {
        const int h = it >> 7, row = (it >> 4) & 7, dc = it & 15;
        float M = -1e30f;
#pragma unroll
        for (int k = 0; k < NW; ++k) M = fmaxf(M, pml[((k * 2 + h) * 8 + row) * 2]);
        float L = 0.f, o[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int k = 0; k < NW; ++k) {
            const float2 v = *(const float2*)&pml[((k * 2 + h) * 8 + row) * 2];
            const float w = exp2f(v.x - M);
            L += w * v.y;
            const float4 a = *(const float4*)&part[((k * 2 + h) * 8 + row) * D + 4 * dc];
            o[0] += w * a.x; o[1] += w * a.y; o[2] += w * a.z; o[3] += w * a.w;
        }
        const float r = vscale / L;
        float s2 = 0.f;
#pragma unroll
        for (int k = 0; k < 4; ++k) { o[k] *= r; s2 += o[k] * o[k]; }
#pragma unroll
        for (int x = 8; x; x >>= 1) s2 += __shfl_xor_sync(0xffffffff, s2, x);
        const float ni = rsqrtf(s2 / D + 1e-5f);
        const long col = (long)b * HQ * D + (h * 8 + row) * D + 4 * dc;
        const uint2 gw = *(const uint2*)(qkv + (long)b * QKV_W + (HQ + 2 * HKV) * D + (h * 8 + row) * D + 4 * dc);
        const float gt[4] = {bf(gw.x & 0xffff), bf(gw.x >> 16), bf(gw.y & 0xffff), bf(gw.y >> 16)};
        __nv_bfloat162 r0 = __floats2bfloat162_rn(o[0] * ni / (1.f + __expf(-gt[0])), o[1] * ni / (1.f + __expf(-gt[1])));
        __nv_bfloat162 r1 = __floats2bfloat162_rn(o[2] * ni / (1.f + __expf(-gt[2])), o[3] * ni / (1.f + __expf(-gt[3])));
        uint2 pk;
        pk.x = *(unsigned*)&r0; pk.y = *(unsigned*)&r1;
        *(uint2*)(out + col) = pk;
    }
}
}  // namespace dnm

template <int PAGE, int NW, int NS>
static void dnm_launch(torch::Tensor qkv, torch::Tensor cs, torch::Tensor kc, torch::Tensor vc, torch::Tensor pt,
                       torch::Tensor seq, torch::Tensor loc, torch::Tensor ks, torch::Tensor vs, torch::Tensor out,
                       double qk_scale) {
    auto f = dnm::decode<PAGE, NW, NS>;
    const int smem = NW * NS * 8192 + 8192;
    static bool set = false;
    if (!set) { cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); set = true; }
    f<<<seq.size(0), NW * 32, smem, c10::cuda::getCurrentCUDAStream()>>>(
        (const unsigned short*)qkv.data_ptr(), cs.data_ptr<float>(), (unsigned char*)kc.data_ptr(),
        (unsigned char*)vc.data_ptr(), pt.data_ptr<int>(), seq.data_ptr<int>(), loc.data_ptr<long>(),
        ks.data_ptr<float>(), vs.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(), (int)pt.size(1), (float)qk_scale);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void dnm_decode(torch::Tensor qkv, torch::Tensor cs, torch::Tensor kc, torch::Tensor vc, torch::Tensor pt,
                torch::Tensor seq, torch::Tensor loc, torch::Tensor ks, torch::Tensor vs, torch::Tensor out, int64_t page,
                double qk_scale, int64_t variant) {
    TORCH_CHECK(page == 1 || page == 64, "dnm: page 1 or 64");
#define DNM_V(NW, NS) page == 1 ? dnm_launch<1, NW, NS>(qkv, cs, kc, vc, pt, seq, loc, ks, vs, out, qk_scale) \
                                : dnm_launch<64, NW, NS>(qkv, cs, kc, vc, pt, seq, loc, ks, vs, out, qk_scale)
    switch (variant) {
        case 0: DNM_V(8, 3); break;
        case 1: DNM_V(8, 2); break;
        case 2: DNM_V(4, 6); break;
        case 3: DNM_V(4, 4); break;
        case 4: DNM_V(4, 3); break;
        default: TORCH_CHECK(false, "dnm variant");
    }
#undef DNM_V
}

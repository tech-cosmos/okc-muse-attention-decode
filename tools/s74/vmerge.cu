// Merge-cost microbenchmark for a native verify-sliding kernel: per (request, KV head) one 16-CTA cluster, each CTA
// holding a 64-row x 64-d FP32 partial in the wgmma m64n64 accumulator layout (+ m, l per row). Reduce-scatter:
// row r goes to CTA r / 4 via st.async (+ mbarrier complete_tx); each owner warp merges one row of 16 partials.
// MODE 0: no merge (just write own rows: launch + cluster floor); 1: full reduce-scatter merge.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <stdint.h>

template <int MODE>
__global__ void __cluster_dims__(16, 1, 1) __launch_bounds__(128) vm(const float* __restrict__ seed, __nv_bfloat16* __restrict__ out) {
    __shared__ __align__(16) float rx[16][4][64];   // [src][row][d]
    __shared__ __align__(16) float rml[16][4][2];
    struct Blk { float x[4][64]; float ml[4][2]; };
    __shared__ __align__(128) Blk stg[MODE >= 2 ? 16 : 1], rcv[MODE >= 2 ? 16 : 1];
    __shared__ __align__(8) unsigned long long mbar;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, c = lane & 3;
    unsigned rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    const unsigned mb = (unsigned)__cvta_generic_to_shared(&mbar);
    if (MODE && tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n\tfence.mbarrier_init.release.cluster;" ::"r"(mb) : "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(mb), "r"(16 * 4 * 66 * 4) : "memory");
    }
    if (MODE) asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    // Fake partial (depends on a global load so it isn't folded).
    const float sd = seed[(blockIdx.x * 128 + tid) & 1023];
    float o[32], m[2], l[2];
#pragma unroll
    for (int i = 0; i < 32; ++i) o[i] = sd * (i + 1);
    m[0] = sd; m[1] = sd * 0.5f; l[0] = 1.f + sd; l[1] = 2.f + sd;
    const int cl = blockIdx.x / 16;
    if (!MODE) {
        if (sd == 12345.f) out[cl] = __float2bfloat16(o[0] + m[0] + l[0] + o[31]);
        return;
    }
    if (MODE >= 2) {
#pragma unroll
        for (int hr = 0; hr < 2; ++hr) {
            const int r = warp * 16 + g + 8 * hr;
#pragma unroll
            for (int j = 0; j < 8; ++j) *(float2*)&stg[r >> 2].x[r & 3][8 * j + 2 * c] = make_float2(o[4 * j + 2 * hr], o[4 * j + 2 * hr + 1]);
            if (c == 0) *(float2*)&stg[r >> 2].ml[r & 3][0] = make_float2(m[hr], l[hr]);
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncthreads();
        asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
        if (MODE == 2 && tid < 16) {
            unsigned src = (unsigned)__cvta_generic_to_shared(&stg[tid]), dst, rmb;
            unsigned lb = (unsigned)__cvta_generic_to_shared(&rcv[rank]);
            asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(dst) : "r"(lb), "r"(tid));
            asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rmb) : "r"(mb), "r"(tid));
            asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                         ::"r"(dst), "r"(src), "n"((int)sizeof(Blk)), "r"(rmb) : "memory");
        }
        if (MODE == 2) asm volatile("{\n\t.reg .pred p;\n\tW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n\t@!p bra W;\n\t}" ::"r"(mb) : "memory");
        float mx = -1e30f;
#pragma unroll
        for (int s = 0; s < 16; ++s) mx = fmaxf(mx, rcv[s].ml[warp][0]);
        float den = 0.f, a0 = 0.f, a1 = 0.f;
#pragma unroll
        for (int s = 0; s < 16; ++s) {
            const float w = exp2f(rcv[s].ml[warp][0] - mx);
            den += w * rcv[s].ml[warp][1];
            const float2 v = *(const float2*)&rcv[s].x[warp][2 * lane];
            a0 += w * v.x; a1 += w * v.y;
        }
        a0 /= den; a1 /= den;
        float s2 = a0 * a0 + a1 * a1;
#pragma unroll
        for (int x = 16; x; x >>= 1) s2 += __shfl_xor_sync(0xffffffff, s2, x);
        const float ni = rsqrtf(s2 / 64 + 1e-5f);
        *(__nv_bfloat162*)(out + ((long)cl * 64 + rank * 4 + warp) * 64 + 2 * lane) = __floats2bfloat162_rn(a0 * ni, a1 * ni);
        return;
    }
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
#pragma unroll
    for (int hr = 0; hr < 2; ++hr) {
        const int r = warp * 16 + g + 8 * hr;     // accumulator row of o[4j + 2hr + {0,1}]
        const unsigned dst_rank = r >> 2, rr = r & 3;
        unsigned base = (unsigned)__cvta_generic_to_shared(&rx[rank][rr][0]);
        unsigned rbase;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rbase) : "r"(base), "r"(dst_rank));
        unsigned rmb;
        asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rmb) : "r"(mb), "r"(dst_rank));
#pragma unroll
        for (int j = 0; j < 8; ++j)
            asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1, %2}, [%3];"
                         ::"r"(rbase + (8 * j + 2 * c) * 4), "f"(o[4 * j + 2 * hr]), "f"(o[4 * j + 2 * hr + 1]), "r"(rmb) : "memory");
        if (c == 0) {
            unsigned mlb = (unsigned)__cvta_generic_to_shared(&rml[rank][rr][0]), rml_r;
            asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(rml_r) : "r"(mlb), "r"(dst_rank));
            asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 [%0], {%1, %2}, [%3];"
                         ::"r"(rml_r), "f"(m[hr]), "f"(l[hr]), "r"(rmb) : "memory");
        }
    }
    // Owner: warp w merges row w of its 4 rows over 16 sources.
    asm volatile("{\n\t.reg .pred p;\n\tW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n\t@!p bra W;\n\t}" ::"r"(mb) : "memory");
    float mx = -1e30f;
#pragma unroll
    for (int s = 0; s < 16; ++s) mx = fmaxf(mx, rml[s][warp][0]);
    float den = 0.f, a0 = 0.f, a1 = 0.f;
#pragma unroll
    for (int s = 0; s < 16; ++s) {
        const float w = exp2f(rml[s][warp][0] - mx);
        den += w * rml[s][warp][1];
        const float2 v = *(const float2*)&rx[s][warp][2 * lane];
        a0 += w * v.x; a1 += w * v.y;
    }
    a0 /= den; a1 /= den;
    float s2 = a0 * a0 + a1 * a1;
#pragma unroll
    for (int x = 16; x; x >>= 1) s2 += __shfl_xor_sync(0xffffffff, s2, x);
    const float ni = rsqrtf(s2 / 64 + 1e-5f);
    *(__nv_bfloat162*)(out + ((long)cl * 64 + rank * 4 + warp) * 64 + 2 * lane) = __floats2bfloat162_rn(a0 * ni, a1 * ni);
}

void vmerge(torch::Tensor seed, torch::Tensor out, int64_t clusters, int64_t mode) {
    auto st = c10::cuda::getCurrentCUDAStream();
    static bool set = false;
    if (!set) { cudaFuncSetAttribute(vm<0>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
                cudaFuncSetAttribute(vm<1>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
                cudaFuncSetAttribute(vm<2>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
                cudaFuncSetAttribute(vm<3>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1); set = true; }
    if (mode == 0) vm<0><<<16 * clusters, 128, 0, st>>>(seed.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr());
    else if (mode == 3) vm<3><<<16 * clusters, 128, 0, st>>>(seed.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr());
    else if (mode == 2) vm<2><<<16 * clusters, 128, 0, st>>>(seed.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr());
    else vm<1><<<16 * clusters, 128, 0, st>>>(seed.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr());
}

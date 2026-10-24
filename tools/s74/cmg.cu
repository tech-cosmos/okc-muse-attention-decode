// Batch-1 decode-full merge as 4-CTA clusters per query head: CTA q merges slots [q S/4, (q+1) S/4) (S/4 <= 128,
// 2 threads per slot, 32 d each), pushes (m, l, acc[64]) to rank 0 with st.async (mbarrier complete_tx); rank 0
// combines, applies v_scale, RMSNorm and sigmoid(gate), stores BF16. Launched with PDL after _gd_full.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace cmg {
constexpr int HQ = 16, HKV = 2, G = 8, D = 64, QKV_W = (2 * HQ + 2 * HKV) * D;
__global__ void __launch_bounds__(256) merge(const unsigned short* __restrict__ qkv, const float* __restrict__ pacc,
                                             const float* __restrict__ pml, const float* __restrict__ vs,
                                             __nv_bfloat16* __restrict__ out, int S) {
    __shared__ float wm[8], red[4][D];
    __shared__ __align__(16) float rcv[4][D + 2];
    __shared__ __align__(8) unsigned long long mbar;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    unsigned rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    const int head = blockIdx.x >> 2, b = blockIdx.y, h = head / G;
    const unsigned mb = (unsigned)__cvta_generic_to_shared(&mbar);
    if (rank == 0 && tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n\tfence.mbarrier_init.release.cluster;" ::"r"(mb) : "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(mb), "r"(4 * (D + 2) * 4) : "memory");
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    float gate = 0.f;
    if (rank == 0 && tid < D) gate = __uint_as_float((unsigned)qkv[(long)b * QKV_W + (HQ + 2 * HKV) * D + head * D + tid] << 16);
    const float vscale = *vs;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    const int per = S >> 2, sl = tid >> 1, half = tid & 1;
    const bool ok = sl < per;
    const long row = ((long)(b * HKV + h) * S + rank * per + sl) * G + head % G;
    float m = -1e30f, l = 0.f;
    float4 a[8];
    if (ok) {
        const float2 ml = *(const float2*)(pml + row * 2);
        m = ml.x; l = ml.y;
        const float4* ap = (const float4*)(pacc + row * D + 32 * half);
#pragma unroll
        for (int i = 0; i < 8; ++i) a[i] = ap[i];
    } else {
#pragma unroll
        for (int i = 0; i < 8; ++i) a[i] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    float mx = m;
#pragma unroll
    for (int x = 16; x; x >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, x));
    if (lane == 0) wm[warp] = mx;
    __syncthreads();
    mx = wm[0];
#pragma unroll
    for (int i = 1; i < 8; ++i) mx = fmaxf(mx, wm[i]);
    const float w = ok ? exp2f(m - mx) : 0.f;
    float lw = half == 0 ? w * l : 0.f;
#pragma unroll
    for (int x = 16; x; x >>= 1) lw += __shfl_xor_sync(0xffffffff, lw, x);
    __shared__ __align__(16) float wa[128][D + 4];
#pragma unroll
    for (int i = 0; i < 8; ++i)
        *(float4*)&wa[sl][32 * half + 4 * i] = make_float4(a[i].x * w, a[i].y * w, a[i].z * w, a[i].w * w);
    __syncthreads();  // all reads of wm (mx) done; wa complete
    if (lane == 0) wm[warp] = lw;
    __shared__ float wacc[4][D];
    {
        const int dd = tid & 63, part = tid >> 6;
        float v = 0.f;
#pragma unroll 8
        for (int i = 0; i < 32; ++i) v += wa[part * 32 + i][dd];
        wacc[part][dd] = v;
    }
    __syncthreads();
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
    if (tid < D + 2) {
        float v = 0.f;
        if (tid < D) {
            v = wacc[0][tid] + wacc[1][tid] + wacc[2][tid] + wacc[3][tid];
        } else if (tid == D) {
            v = mx;
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) v += wm[i];
        }
        unsigned dst = (unsigned)__cvta_generic_to_shared(&rcv[rank][tid]), rdst, rmb;
        asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(rdst) : "r"(dst));
        asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(rmb) : "r"(mb));
        asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 [%0], %1, [%2];"
                     ::"r"(rdst), "r"(__float_as_uint(v)), "r"(rmb) : "memory");
    }
    if (rank != 0) return;
    asm volatile("{\n\t.reg .pred p;\n\tW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n\t@!p bra W;\n\t}" ::"r"(mb) : "memory");
    if (warp < 2) {
        const float M = fmaxf(fmaxf(rcv[0][D], rcv[1][D]), fmaxf(rcv[2][D], rcv[3][D]));
        float den = 0.f, o = 0.f;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const float wq = exp2f(rcv[q][D] - M);
            den += wq * rcv[q][D + 1];
            o += wq * rcv[q][tid];
        }
        o = o / den * vscale;
        float s2 = o * o;
#pragma unroll
        for (int x = 16; x; x >>= 1) s2 += __shfl_xor_sync(0xffffffff, s2, x);
        if (lane == 0) red[0][warp] = s2;
        asm volatile("bar.sync 1, 64;" ::: "memory");
        s2 = red[0][0] + red[0][1];
        o *= rsqrtf(s2 / D + 1e-5f) / (1.f + __expf(-gate));
        out[((long)b * HQ + head) * D + tid] = __float2bfloat16(o);
    }
}
}  // namespace cmg

void cmg_merge(torch::Tensor qkv, torch::Tensor pacc, torch::Tensor pml, torch::Tensor vs, torch::Tensor out, int64_t S) {
    int batch = out.size(0);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(4 * 16, batch);
    cfg.blockDim = dim3(256);
    cfg.stream = c10::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at[2];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = 4; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    at[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[1].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at; cfg.numAttrs = 2;
    cudaLaunchKernelEx(&cfg, cmg::merge, (const unsigned short*)qkv.data_ptr(), pacc.data_ptr<float>(), pml.data_ptr<float>(),
                       vs.data_ptr<float>(), (__nv_bfloat16*)out.data_ptr(), (int)S);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

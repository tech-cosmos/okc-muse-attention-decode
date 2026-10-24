// Cost of merging split partials inside one kernel via a thread-block cluster (DSMEM), vs not merging.
// Each CTA = one split: writes a [16 rows x 64] fp32 partial + m/l to its shared memory; with MERGE, after a
// cluster barrier each CTA merges 16/NC rows... here rank r merges row slice r (NC <= 16) reading all peers.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
template <int NC, int MERGE>
__global__ void __launch_bounds__(128) kmerge(const float* __restrict__ in, float* __restrict__ out) {
    __shared__ float acc[16][64];
    __shared__ float ml[16][2];
    cg::cluster_group cl = cg::this_cluster();
    int rank = cl.block_rank();
    int t = threadIdx.x;
    // "partial": a cold global read (stands in for the split's attention result)
    const float* src = in + (size_t)blockIdx.x * 16 * 66;
    for (int i = t; i < 16 * 64; i += 128) acc[i / 64][i % 64] = src[i];
    if (t < 16) { ml[t][0] = src[1024 + t]; ml[t][1] = src[1040 + t] + 1.f; }
    if (MERGE == 8) {
        asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
        __syncthreads();
        float* dst = out + (size_t)blockIdx.x * 1056;
        for (int i = t; i < 16 * 64; i += 128) dst[i] = acc[i / 64][i % 64];
        if (t < 16) { dst[1024 + t] = ml[t][0]; dst[1040 + t] = ml[t][1]; }
        return;
    }
    if (!MERGE) {
        __syncthreads();
        for (int i = t; i < 16 * 64; i += 128) out[(size_t)blockIdx.x * 1024 + i] = acc[i / 64][i % 64] / ml[i / 64][1];
        return;
    }
    if (MERGE == 4 || MERGE == 6 || MERGE == 7) {
        // push: rows of slice o go to CTA o's receive buffer at [rank]
        constexpr int RPC = 16 / NC;
        __shared__ float racc[NC][RPC][64];
        __shared__ float rml[NC][RPC][2];
        __syncthreads();
        for (int i = t; i < 16 * 64; i += 128) {
            int row = i / 64, col = i % 64, o = row / RPC;
            cl.map_shared_rank(&racc[0][0][0], o)[(rank * RPC + row % RPC) * 64 + col] = acc[row][col];
        }
        if (t < 16) {
            int o = t / RPC;
            float* p = cl.map_shared_rank(&rml[0][0][0], o) + (rank * RPC + t % RPC) * 2;
            p[0] = ml[t][0]; p[1] = ml[t][1];
        }
        if (MERGE == 6) __syncthreads();
        else if (MERGE == 7) asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
        else cl.sync();
        for (int i = t; i < RPC * 64; i += 128) {
            int r = i / 64, col = i % 64;
            float m = -1e30f, ms[NC], as[NC], lsum[NC];
#pragma unroll
            for (int s = 0; s < NC; ++s) { ms[s] = rml[s][r][0]; lsum[s] = rml[s][r][1]; as[s] = racc[s][r][col]; }
            if (MERGE == 4) {
#pragma unroll
                for (int s = 0; s < NC; ++s) m = fmaxf(m, ms[s]);
                float a = 0.f, l = 0.f;
#pragma unroll
                for (int s = 0; s < NC; ++s) { float w = exp2f(ms[s] - m); a += w * as[s]; l += w * lsum[s]; }
                out[(size_t)(blockIdx.x / NC) * 1024 + (rank * RPC + r) * 64 + col] = __fdividef(a, l);
                continue;
            }
#pragma unroll
            for (int s = 0; s < NC; ++s) m = fmaxf(m, rml[s][r][0]);
            float a = 0.f, l = 0.f;
#pragma unroll
            for (int s = 0; s < NC; ++s) { float w = exp2f(rml[s][r][0] - m); a += w * racc[s][r][col]; l += w * rml[s][r][1]; }
            out[(size_t)(blockIdx.x / NC) * 1024 + (rank * RPC + r) * 64 + col] = a / l;
        }
        return;
    }
    if (MERGE == 3) { cl.sync(); cl.sync(); if (t == 0) out[blockIdx.x] = acc[0][0]; return; }
    if (MERGE == 2) __syncthreads(); else cl.sync();
    // rank r merges rows [r * 16 / NC, (r + 1) * 16 / NC)
    constexpr int RPC = 16 / NC;
    for (int i = t; i < RPC * 64; i += 128) {
        int row = rank * RPC + i / 64, col = i % 64;
        float m = -1e30f;
        for (int s = 0; s < NC; ++s) m = fmaxf(m, cl.map_shared_rank(&ml[0][0], s)[row * 2]);
        float a = 0.f, l = 0.f;
        for (int s = 0; s < NC; ++s) {
            float* pm = cl.map_shared_rank(&ml[0][0], s);
            float w = exp2f(pm[row * 2] - m);
            a += w * cl.map_shared_rank(&acc[0][0], s)[row * 64 + col];
            l += w * pm[row * 2 + 1];
        }
        out[(size_t)(blockIdx.x / NC) * 1024 + row * 64 + col] = a / l;
    }
    if (MERGE != 2) cl.sync();  // peers' smem must stay alive until everyone has read it
}
// Production-style alternative: each split CTA stores its partial to global (MERGE 8 in kmerge), then a separate
// merge kernel launched with programmatic dependent launch (PDL) merges NC splits per group.
template <int NC>
__global__ void __launch_bounds__(1024) gmerge(const float* __restrict__ part, float* __restrict__ out) {
    int t = threadIdx.x, r = t / 64, col = t % 64;
    asm volatile("griddepcontrol.wait;\n" ::: "memory");
    const float* p = part + (size_t)blockIdx.x * NC * 16 * 66;
    float ms[NC], ls[NC], as[NC], m = -1e30f;
#pragma unroll
    for (int s = 0; s < NC; ++s) { ms[s] = p[s * 1056 + 1024 + r]; ls[s] = p[s * 1056 + 1040 + r]; as[s] = p[s * 1056 + r * 64 + col]; }
#pragma unroll
    for (int s = 0; s < NC; ++s) m = fmaxf(m, ms[s]);
    float a = 0.f, l = 0.f;
#pragma unroll
    for (int s = 0; s < NC; ++s) { float w = exp2f(ms[s] - m); a += w * as[s]; l += w * ls[s]; }
    out[(size_t)blockIdx.x * 1024 + t] = __fdividef(a, l);
}
template <int NC, int MERGE>
void launch(torch::Tensor in, torch::Tensor out, int grid) {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(grid); cfg.blockDim = dim3(128); cfg.dynamicSmemBytes = 0;
    cfg.stream = c10::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeClusterDimension;
    at[0].val.clusterDim.x = NC; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1;
    cfg.attrs = at; cfg.numAttrs = 1;
    if (NC > 8) cudaFuncSetAttribute(kmerge<NC, MERGE>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    cudaLaunchKernelEx(&cfg, kmerge<NC, MERGE>, (const float*)in.data_ptr<float>(), out.data_ptr<float>());
}
template <int NC>
void launch_sep(torch::Tensor in, torch::Tensor out, torch::Tensor part, int grid) {
    launch<1, 8>(in, part, grid);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(grid / NC); cfg.blockDim = dim3(1024); cfg.dynamicSmemBytes = 0;
    cfg.stream = c10::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute at[1];
    at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; at[0].val.programmaticStreamSerializationAllowed = 1;
    cfg.attrs = at; cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, gmerge<NC>, (const float*)part.data_ptr<float>(), out.data_ptr<float>());
}
void run_sep(int nc, torch::Tensor in, torch::Tensor out, torch::Tensor part, int grid) {
    if (nc == 4) launch_sep<4>(in, out, part, grid);
    if (nc == 16) launch_sep<16>(in, out, part, grid);
}
void run(int nc, int merge, torch::Tensor in, torch::Tensor out, int grid) {
    if (merge == 6) { launch<1, 6>(in, out, grid); return; }
    if (merge == 7) { if (nc == 1) launch<1, 7>(in, out, grid); if (nc == 16) launch<16, 7>(in, out, grid); return; }
    if (merge == 4) { if (nc == 1) launch<1, 4>(in, out, grid); if (nc == 4) launch<4, 4>(in, out, grid);
                      if (nc == 8) launch<8, 4>(in, out, grid); if (nc == 16) launch<16, 4>(in, out, grid); return; }
    if (nc == 1) { if (merge == 2) launch<1, 2>(in, out, grid); else if (merge == 3) launch<1, 3>(in, out, grid); else merge ? launch<1, 1>(in, out, grid) : launch<1, 0>(in, out, grid); }
    if (nc == 2) { if (merge == 3) launch<2, 3>(in, out, grid); }
    if (merge == 4) { if (nc == 1) launch<1, 4>(in, out, grid); if (nc == 4) launch<4, 4>(in, out, grid);
                      if (nc == 8) launch<8, 4>(in, out, grid); if (nc == 16) launch<16, 4>(in, out, grid); return; }
    if (nc == 4) { merge ? launch<4, 1>(in, out, grid) : launch<4, 0>(in, out, grid); }
    if (nc == 8 && merge == 3) { launch<8, 3>(in, out, grid); return; }
    if (nc == 8) { merge ? launch<8, 1>(in, out, grid) : launch<8, 0>(in, out, grid); }
    if (nc == 16) { merge ? launch<16, 1>(in, out, grid) : launch<16, 0>(in, out, grid); }
}

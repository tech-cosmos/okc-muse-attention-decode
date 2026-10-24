// Sliding-window gather floor: each CTA copies KPC keys' K and V rows (128 B each, random slots via the page
// table) into shared memory with every cp.async in flight at once, then touches the data.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <stdint.h>
__global__ void sgather(const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc, const int32_t* __restrict__ pt,
                        const int32_t* __restrict__ seq, int max_pages, int kpc, int window, float* __restrict__ out) {
    extern __shared__ __align__(128) uint8_t sm[];
    int cpr = window / kpc;              // CTAs per request
    int b = blockIdx.x / cpr, part = blockIdx.x % cpr;
    int s = seq[b];
    int lo = max(s - window, 0) + part * kpc;
    const int32_t* ptr = pt + (int64_t)b * max_pages;
    int nchunks = kpc * 8;               // 16-byte chunks per tensor
    for (int c = threadIdx.x; c < nchunks; c += blockDim.x) {
        int key = lo + c / 8, off = (c % 8) * 16;
        int64_t slot = ptr[min(key, s - 1)];
        uint32_t dk = (uint32_t)__cvta_generic_to_shared(sm + c * 16);
        uint32_t dv = (uint32_t)__cvta_generic_to_shared(sm + nchunks * 16 + c * 16);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dk), "l"(kc + slot * 128 + off));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dv), "l"(vc + slot * 128 + off));
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();
    float acc = 0.f;
    for (int c = threadIdx.x; c < nchunks * 2; c += blockDim.x) acc += (float)sm[c * 16];
    if (acc == 1234.5f) out[blockIdx.x] = acc;
}
void run(torch::Tensor kc, torch::Tensor vc, torch::Tensor pt, torch::Tensor seq, int kpc, int window, int threads,
         torch::Tensor out) {
    int batch = seq.size(0);
    int grid = batch * (window / kpc);
    int smem = kpc * 256;
    cudaFuncSetAttribute(sgather, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    sgather<<<grid, threads, smem, c10::cuda::getCurrentCUDAStream()>>>(kc.data_ptr<uint8_t>(), vc.data_ptr<uint8_t>(),
        pt.data_ptr<int32_t>(), seq.data_ptr<int32_t>(), pt.size(1), kpc, window, out.data_ptr<float>());
}

// Gather ceiling with per-row bulk copies (TMA engine): one producer warp per CTA issues
// cp.async.bulk (128 B) for every K and V row of a tile into an mbarrier ring; a consumer warp waits and frees.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <stdint.h>

template <int BN, int NS, int AH>
__global__ void __launch_bounds__(64) tg(const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
                                         const int* __restrict__ idx, float* __restrict__ out, int chunk) {
    extern __shared__ __align__(1024) uint8_t sm[];
    __shared__ __align__(8) unsigned long long full[NS], empty[NS];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (threadIdx.x == 0) {
        for (int s = 0; s < NS; ++s) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"((unsigned)__cvta_generic_to_shared(&full[s])));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"((unsigned)__cvta_generic_to_shared(&empty[s])));
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    const long start = (long)blockIdx.x * chunk;
    const int tiles = chunk / BN;
    constexpr int RPL = BN / 32;  // rows per lane
    if (warp == 0) {
        __shared__ int ix[AH + 1][BN];
        auto stage_idx = [&](int t) {
            if (t < tiles) {
#pragma unroll
                for (int r = 0; r < RPL; ++r)
                    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"((unsigned)__cvta_generic_to_shared(&ix[t % (AH + 1)][r * 32 + lane])),
                                 "l"(idx + start + (long)t * BN + r * 32 + lane) : "memory");
            }
            asm volatile("cp.async.commit_group;" ::: "memory");
        };
#pragma unroll
        for (int a = 0; a < AH; ++a) stage_idx(a);
        for (int t = 0; t < tiles; ++t) {
            const int s = t % NS, ph = (t / NS) & 1;
            stage_idx(t + AH);
            asm volatile("cp.async.wait_group %0;" ::"n"(AH) : "memory");
            __syncwarp();
            const unsigned eb = (unsigned)__cvta_generic_to_shared(&empty[s]);
            const unsigned fb = (unsigned)__cvta_generic_to_shared(&full[s]);
            if (t >= NS) {
                asm volatile("{\n\t.reg .pred p;\n\tW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t@!p bra W;\n\t}"
                             ::"r"(eb), "r"(ph ^ 1) : "memory");
            }
            if (lane == 0) asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(fb), "r"(BN * 256) : "memory");
            __syncwarp();
            const unsigned dk = (unsigned)__cvta_generic_to_shared(sm + s * BN * 256);
#pragma unroll
            for (int r = 0; r < RPL; ++r) {
                const int row = r * 32 + lane;
                const long g = (long)ix[t % (AH + 1)][row] * 128;
                asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 128, [%2];"
                             ::"r"(dk + row * 128), "l"(kc + g), "r"(fb) : "memory");
                asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 128, [%2];"
                             ::"r"(dk + BN * 128 + row * 128), "l"(vc + g), "r"(fb) : "memory");
            }
__syncwarp();
        }
    } else {
        float acc = 0.f;
        for (int t = 0; t < tiles; ++t) {
            const int s = t % NS, ph = (t / NS) & 1;
            const unsigned fb = (unsigned)__cvta_generic_to_shared(&full[s]);
            asm volatile("{\n\t.reg .pred p;\n\tW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t@!p bra W;\n\t}"
                         ::"r"(fb), "r"(ph) : "memory");
            acc += (float)sm[s * BN * 256 + lane * 4];
            __syncwarp();
            if (lane == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"((unsigned)__cvta_generic_to_shared(&empty[s])) : "memory");
        }
        if (lane == 0) out[blockIdx.x] = acc;
    }
}

void tgather(torch::Tensor kc, torch::Tensor vc, torch::Tensor idx, torch::Tensor out, int64_t chunk, int64_t bn, int64_t ns) {
    int grid = idx.size(0) / chunk;
    auto st = c10::cuda::getCurrentCUDAStream();
#define L(B, N) if (bn == B && ns == N) { auto f = tg<B, N, 4>; int smem = N * B * 256; \
    cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
    f<<<grid, 64, smem, st>>>(kc.data_ptr<uint8_t>(), vc.data_ptr<uint8_t>(), idx.data_ptr<int>(), out.data_ptr<float>(), (int)chunk); return; }
    L(32, 4) L(32, 8) L(64, 4) L(64, 6) L(64, 8) L(128, 3) L(128, 4)
    TORCH_CHECK(false, "no variant");
}

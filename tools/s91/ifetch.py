"""s91: instruction-fetch cost after an L2 flush: straight-line N FFMAs vs a loop of the same count, one warp,
%globaltimer inside. python tools/s91/ifetch.py"""
import os, sys, fcntl
import torch
from torch.utils.cpp_extension import load_inline
src = r'''
#include <torch/extension.h>
template <int N, bool LOOP>
__global__ void k(float* o, unsigned long long* t) {
    float a = o[0], b = o[1], c = o[2], d = o[3];
    unsigned long long t0, t1;
    if (a == 12345.f) t0 = 1; else asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    if (LOOP) {
        #pragma unroll 1
        for (int i = 0; i < N / 8; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(b), "f"(c));
                asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(b) : "f"(c), "f"(d));
                asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(c) : "f"(d), "f"(a));
                asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(d) : "f"(a), "f"(b));
            }
        }
    } else {
        #pragma unroll
        for (int i = 0; i < N / 4; ++i) {
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(b), "f"(c));
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(b) : "f"(c), "f"(d));
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(c) : "f"(d), "f"(a));
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(d) : "f"(a), "f"(b));
        }
    }
    if (a + b + c + d == 12345.f) t1 = 0; else asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
    if (threadIdx.x == 0) { t[blockIdx.x] = t1 - t0; o[4 + blockIdx.x] = a + b + c + d; }
}
void run(torch::Tensor o, torch::Tensor t, int64_t n, int64_t loop, int64_t ctas) {
#define L(N) if (n == N) { if (loop) k<N, true><<<ctas, 32>>>(o.data_ptr<float>(), (unsigned long long*)t.data_ptr()); \
                           else k<N, false><<<ctas, 32>>>(o.data_ptr<float>(), (unsigned long long*)t.data_ptr()); }
    L(512) L(1024) L(2048) L(4096)
}
'''
m = load_inline("s91_ifetch2", cpp_sources="void run(torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);", cuda_sources=src,
                functions=["run"], extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"])
o = torch.ones(1024, device="cuda") * 0.999; t = torch.zeros(256, dtype=torch.int64, device="cuda")
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda"); sink = torch.zeros((), dtype=torch.int64, device="cuda")
with open("/tmp/okc-gpu.lock", "w") as lk:
    fcntl.flock(lk, fcntl.LOCK_EX)
    for n in (512, 1024, 2048, 4096):
        for loop in (0, 1):
            for cold in (0, 1):
                r = []
                for _ in range(30):
                    if cold: torch.sum(flush.view(torch.int64), 0, out=sink)
                    m.run(o, t, n, loop, 16); torch.cuda.synchronize()
                    r.append(t[:16].float().mean().item())
                r.sort()
                print(f"N {n:5d} {'loop ' if loop else 'line '} {'cold' if cold else 'warm'}: {r[15] / 1000:.3f} us", flush=True)

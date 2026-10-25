// Pipe-sharing microbench: MUFU.EX2 vs F2FP (f32x2->f16x2 pack, e4m3x2->f16x2 unpack, f32x2->e4m3x2) vs ALU ops.
#include <cstdio>
#include <cuda_fp16.h>
#define N 4096
template <int mode> __global__ void k( float* out, unsigned* outu, float seed) {
  float a[8]; unsigned u[8];
  for (int i = 0; i < 8; i++) { a[i] = seed * (threadIdx.x + i); u[i] = __float_as_uint(a[i]) ^ (i * 0x01010101u); }
  for (int it = 0; it < N; it++) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      if (mode & 1) asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(a[i]));
      if (mode & 2) { unsigned r; asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(a[i]), "f"(a[(i+1)&7])); u[i] ^= r; }
      if (mode & 4) { unsigned r; unsigned short s = (unsigned short)u[i]; asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(s)); u[i] += r; }
      if (mode & 8) { unsigned short r; asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(a[i]), "f"(a[(i+1)&7])); u[i] ^= r; }
      if (mode & 16) { unsigned r; asm volatile("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(r) : "r"(u[i]), "r"(u[(i+1)&7])); u[i] = r + 1; }
      if (mode & 32) { asm volatile("fma.rn.f32 %0, %0, %1, %1;" : "+f"(a[i]) : "f"(seed)); }
    }
  }
  float s = 0; unsigned t = 0;
  for (int i = 0; i < 8; i++) { s += a[i]; t ^= u[i]; }
  out[blockIdx.x * blockDim.x + threadIdx.x] = s; outu[blockIdx.x * blockDim.x + threadIdx.x] = t;
}
int main() {
  float* o; unsigned* ou; cudaMalloc(&o, 132 * 8 * 1024 * 4); cudaMalloc(&ou, 132 * 8 * 1024 * 4);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  int modes[] = {1, 2, 4, 8, 16, 32, 3, 5, 9, 17, 33, 6, 7, 13};
  const char* names[] = {"ex2", "pack f16x2", "unpack e4m3->f16x2", "pack e4m3x2", "prmt", "fma", "ex2+pack", "ex2+unpack", "ex2+pack8", "ex2+prmt", "ex2+fma", "pack+unpack", "ex2+pack+unpack", "ex2+unpack+pack8"};
  for (int m = 0; m < 14; m++) {
    for (int rep = 0; rep < 2; rep++) {
      cudaEventRecord(e0);
      switch (modes[m]) {
#define C(M) case M: k<M><<<132 * 4, 512>>>(o, ou, 1e-3f); break;
        C(1) C(2) C(4) C(8) C(16) C(32) C(3) C(5) C(9) C(17) C(33) C(6) C(7) C(13)
      }
      cudaEventRecord(e1); cudaEventSynchronize(e1);
      float ms; cudaEventElapsedTime(&ms, e0, e1);
      // ops per SM per clock: total warp-instr per kind
      double ops = 132.0 * 4 * 512 * N * 8;  // per kind (thread-level)
      if (rep) printf("%-20s %8.3f ms  %.1f thread-ops/clk/SM per kind (at 1.755 GHz)\n", names[m], ms, ops / (ms * 1e-3) / 132 / 1.755e9);
    }
  }
}

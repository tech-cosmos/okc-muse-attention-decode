// s122: can Hopper overlap wgmma (tensor) with MUFU.EX2 / FFMA from other warpgroups on the same SM?
// WG0 = tensor loop (wgmma m64n64k16 f16, SS or RS like _gw_verify's P@V), WG1 = MUFU / FFMA / softmax-mix loop.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#define D32 "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31])
__device__ __forceinline__ void wg_ss(float (&d)[32], uint64_t da, uint64_t db) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\nwgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1, 0, 1;\n}\n"
               : D32 : "l"(da), "l"(db));
}
__device__ __forceinline__ void wg_rs(float (&d)[32], const uint32_t (&a)[4], uint64_t db) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\nwgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, {%32,%33,%34,%35}, %36, p, 1, 1, 1;\n}\n"
               : D32 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db));
}
__device__ __forceinline__ void wg_ss8(float (&d)[32], uint64_t da, uint64_t db) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, 1, 0;\nwgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, %32, %33, p, 1, 1;\n}\n"
               : D32 : "l"(da), "l"(db));
}
__device__ __forceinline__ uint64_t desc(const void* p, int lbo, int sbo) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
  return (uint64_t)((a & 0x3FFFF) >> 4) | ((uint64_t)(lbo >> 4) << 16) | ((uint64_t)(sbo >> 4) << 32);
}
// tmode: 0 none, 1 SS f16, 2 RS f16 (P@V-like: 4 x k16), 3 QK-like (2 x SS e4m3 k32) + PV-like (4 x RS k16)
// smode: 0 none, 1 ex2, 2 ffma, 3 softmax mix (ffma + ex2 + fadd + 0.5 cvt per element)
template <int tmode, int smode> __global__ void __launch_bounds__(256) k(int nt, int ns, float* out, long long* cyc) {
  __shared__ __align__(1024) unsigned char sm[32768];
  for (int i = threadIdx.x; i < 32768 / 4; i += blockDim.x) ((uint32_t*)sm)[i] = 0x3c003c00u * (i & 1);
  __syncthreads();
  int wg = threadIdx.x / 128;
  long long t0 = clock64();
  float res = 0.f;
  if (wg == 0 && tmode) {
    float d[32]; for (int i = 0; i < 32; i++) d[i] = 0.f;
    uint32_t a[4] = {0x3c003c00u, 0x3c003c00u, 0x3c003c00u, 0x3c003c00u};
    uint64_t da = desc(sm, 128, 256), db = desc(sm + 16384, 128, 256);
    for (int it = 0; it < nt; it++) {
      asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
      if (tmode == 1) { for (int j = 0; j < 4; j++) wg_ss(d, da + 2 * j, db + 2 * j); }
      else if (tmode == 2) { for (int j = 0; j < 4; j++) wg_rs(d, a, db + 2 * j); }
      else { wg_ss8(d, da, db); wg_ss8(d, da + 2, db + 2);
             asm volatile("wgmma.commit_group.sync.aligned;\nwgmma.wait_group.sync.aligned 0;\n" ::: "memory");
             asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
             for (int j = 0; j < 4; j++) wg_rs(d, a, db + 2 * j); }
      asm volatile("wgmma.commit_group.sync.aligned;\nwgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    }
    for (int i = 0; i < 32; i++) res += d[i];
  } else if (wg == 1 && smode) {
    float x[16]; for (int i = 0; i < 16; i++) x[i] = 0.001f * (threadIdx.x + i);
    float s = 0.f; uint32_t u = 0;
    for (int it = 0; it < ns; it++) {
#pragma unroll
      for (int i = 0; i < 16; i++) {
        if (smode == 1) asm volatile("{.reg .f32 t; neg.f32 t, %0; ex2.approx.ftz.f32 %0, t;}" : "+f"(x[i]));
        else if (smode == 2) asm volatile("fma.rn.f32 %0, %0, 0f3F000000, 0f3E800000;" : "+f"(x[i]));
        else {
          float y; asm volatile("fma.rn.f32 %0, %1, 0fBF000000, 0f3E800000;" : "=f"(y) : "f"(x[i]));
          asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(x[i]) : "f"(y));
          asm volatile("add.f32 %0, %0, %1;" : "+f"(s) : "f"(x[i]));
          if (i & 1) { uint32_t r; asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(x[i]), "f"(x[i - 1])); u ^= r; }
        }
      }
    }
    for (int i = 0; i < 16; i++) res += x[i];
    res += s + (float)u;
  }
  long long t1 = clock64();
  out[blockIdx.x * blockDim.x + threadIdx.x] = res;
  if (threadIdx.x % 128 == 0) cyc[blockIdx.x * 2 + wg] = t1 - t0;
}

template <int T, int S> void L(int g, int nt, int ns, float* o, long long* c) { k<T, S><<<g, 256>>>(nt, ns, o, c); }
typedef void (*LF)(int, int, int, float*, long long*);
static LF tab[4][4] = {{L<0,0>,L<0,1>,L<0,2>,L<0,3>},{L<1,0>,L<1,1>,L<1,2>,L<1,3>},{L<2,0>,L<2,1>,L<2,2>,L<2,3>},{L<3,0>,L<3,1>,L<3,2>,L<3,3>}};
void launch(int t, int s, int g, int nt, int ns, float* o, long long* c) { tab[t][s](g, nt, ns, o, c); }
int main(int argc, char** argv) {
  int nt = atoi(argv[1]), ns = atoi(argv[2]), grid = argc > 3 ? atoi(argv[3]) : 132;
  float* o; long long* c; cudaMalloc(&o, 1024 * 256 * 4); cudaMalloc(&c, 1024 * 2 * 8);
  long long* hc = (long long*)malloc(1024 * 2 * 8);
  cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  int cases[][2] = {{1,0},{2,0},{3,0},{0,1},{0,2},{0,3},{1,1},{2,1},{3,1},{1,2},{2,2},{3,2},{1,3},{2,3},{3,3}};
  for (auto& cs : cases) {
    float best = 1e9;
    for (int r = 0; r < 5; r++) {
      cudaEventRecord(e0); launch(cs[0], cs[1], grid, nt, ns, o, c); cudaEventRecord(e1); cudaEventSynchronize(e1);
      float ms; cudaEventElapsedTime(&ms, e0, e1); if (ms < best) best = ms;
    }
    cudaMemcpy(hc, c, grid * 2 * 8, cudaMemcpyDeviceToHost);
    double a0 = 0, a1 = 0; for (int b = 0; b < grid; b++) { a0 += hc[2 * b]; a1 += hc[2 * b + 1]; }
    printf("tmode %d smode %d: %8.1f us   WG0 %9.0f clk  WG1 %9.0f clk\n", cs[0], cs[1], best * 1e3, a0 / grid, a1 / grid);
  }
  cudaError_t e = cudaGetLastError(); if (e) printf("err %s\n", cudaGetErrorString(e));
}

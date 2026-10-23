#include <cuda_runtime.h>
__global__ void probe(float* dst,const float* src) {
 asm volatile("griddepcontrol.wait;":::"memory");
 dst[threadIdx.x]=src[threadIdx.x];
}
__global__ void probe_intrinsic(float* dst,const float* src) {
 cudaGridDependencySynchronize();
 dst[threadIdx.x]=src[threadIdx.x];
}

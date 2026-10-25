import sys
s=open(sys.argv[1]).read()
s=s.replace("template <int PAGE>\n__global__", """__device__ unsigned long long g_dbg[64][2][8];
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
#define STAMP(k) do { if (ABL == 9 && lane == 0 && ((rank == 15 && wk == 3) || (rank == 0 && wk == 0))) g_dbg[ci][rank == 15][k] = gtime(); } while (0)
template <int PAGE>
__global__""",1)
s=s.replace("    const unsigned mb = (unsigned)__cvta_generic_to_shared(&mbar);\n","    const unsigned mb = (unsigned)__cvta_generic_to_shared(&mbar);\n    STAMP(0);\n",1)
s=s.replace("#undef CSL_LOADS\n","#undef CSL_LOADS\n    STAMP(1);\n",1)
s=s.replace("    // S = Q K^T: tile j","    STAMP(2);\n    // S = Q K^T: tile j",1)
s=s.replace("    const float moa = ma - 8.f","    STAMP(3);\n    const float moa = ma - 8.f",1)
s=s.replace("    __syncthreads();\n    {\n        const int row = tid >> 3","    STAMP(4);\n    __syncthreads();\n    {\n        const int row = tid >> 3",1)
s=s.replace("        asm volatile(\"barrier.cluster.wait.aligned;\" ::: \"memory\");\n        const unsigned owner = row;","        asm volatile(\"barrier.cluster.wait.aligned;\" ::: \"memory\");\n        STAMP(5);\n        const unsigned owner = row;",1)
s=s.replace("        float M = -1e30f;\n#pragma unroll\n        for (int s = 0; s < 16; ++s) M = fmaxf(M, rml[s][0]);","        STAMP(6);\n        float M = -1e30f;\n#pragma unroll\n        for (int s = 0; s < 16; ++s) M = fmaxf(M, rml[s][0]);",1)
s=s.replace("__floats2bfloat162_rn(o0, o1);\n    }\n}","__floats2bfloat162_rn(o0, o1);\n        STAMP(7);\n    }\n}",1)
s=s.replace("void vtk_verify(", "void vtk_dbg(torch::Tensor d) { cudaMemcpyFromSymbol(d.data_ptr(), vtk::g_dbg, sizeof(vtk::g_dbg)); }\nvoid vtk_verify(")
print(s.count("STAMP("))
open(sys.argv[2],"w").write(s)

import os, sys, torch
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
from torch.utils.cpp_extension import load_inline
here = os.path.dirname(os.path.abspath(__file__))
def build(ns=3, pregs=40, cregs=96, lazy=1, tag=""):
    abl = int(os.environ.get("NV_ABL", "0")); lag = int(os.environ.get("NV_LAG", "1")); pfd = int(os.environ.get("NV_PFD", "4")); tr = int(os.environ.get("NV_TR", "1")); tag += f"a{abl}g{lag}p{pfd}t{tr}"
    src = open(os.path.join(here, "nverify.cu")).read()
    return load_inline(f"nverify_s71_{ns}_{pregs}_{cregs}_{lazy}{tag}",
        cpp_sources="void nverify(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t, int64_t, int64_t);",
        cuda_sources=src, functions=["nverify"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", f"-DNV_NS={ns}", f"-DNV_PREGS={pregs}",
                           f"-DNV_CREGS={cregs}", f"-DNV_LAZY={lazy}", f"-DNV_ABL={abl}", f"-DNV_LAG={lag}", f"-DNV_PFD={pfd}", f"-DNV_TR={tr}", "-lineinfo", "-Xptxas=-v"], verbose=False)
if __name__ == "__main__":
    build(*[int(a) for a in sys.argv[1:]])

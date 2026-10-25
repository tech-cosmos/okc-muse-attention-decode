import os, sys, torch
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
from torch.utils.cpp_extension import load_inline
here = os.path.dirname(os.path.abspath(__file__))
def build(extra=(), tag="x", src="nvl.cu"):
    s = open(os.path.join(here, src)).read()
    return load_inline("nvl_s78_" + tag, cpp_sources="void nverify(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t, int64_t, int64_t);",
        cuda_sources=s, functions=["nverify"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-DNV_NS=3", "-DNV_PREGS=32", "-DNV_CREGS=104", "-DNV_LAG=1", "-DNV_PFD=4", "-DNV_TR=1", "-Xptxas=-v"] + list(extra), verbose=True)
if __name__ == "__main__":
    build(sys.argv[2:], sys.argv[1])

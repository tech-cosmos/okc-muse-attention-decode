import os, sys
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
from torch.utils.cpp_extension import load_inline
here = os.path.dirname(os.path.abspath(__file__))
def build(ns=4, pregs=32, cregs=104, lag=1, pfd=4, pk=4, pipe=1, unr=1, abl=0, minb=2, ver=2, cvp=2, wab=0):
    src = open(os.path.join(here, "nvf.cu" if ver == 1 else f"nvf{ver}.cu")).read()
    return load_inline(f"nvf_s111_{ns}_{pregs}_{cregs}_{lag}_{pfd}_{pk}_{pipe}_{unr}_{abl}_{minb}_v{ver}_{cvp}_{wab}",
        cpp_sources="void nvf(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t, int64_t, int64_t); int64_t nvf_occ();" if ver >= 2 else "void nvf(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t, int64_t, int64_t);",
        cuda_sources=src, functions=["nvf", "nvf_occ"] if ver >= 2 else ["nvf"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", f"-DNV_NS={ns}", f"-DNV_PREGS={pregs}",
                           f"-DNV_CREGS={cregs}", f"-DNV_LAG={lag}", f"-DNV_PFD={pfd}", f"-DNV_PK={pk}", f"-DNV_PIPE={pipe}", f"-DNV_UNR={unr}", f"-DNV_ABL={abl}", f"-DNV_MINB={minb}", f"-DNV_CVP={cvp}", f"-DNV_WAB={wab}",
                           "-lineinfo"] + ([]) + [ "-Xptxas=-v"], verbose=os.environ.get("V") == "1")
if __name__ == "__main__":
    build(*[int(a) for a in sys.argv[1:]])

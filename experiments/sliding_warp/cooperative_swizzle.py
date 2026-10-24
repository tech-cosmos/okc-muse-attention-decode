from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline
ROOT=Path(__file__).resolve().parent
BUILD=ROOT/'.build-cooperative_swizzle';BUILD.mkdir(exist_ok=True)
module=load_inline(name='codex_sliding_mma_cooperative_swizzle',build_directory=str(BUILD),
 cpp_sources=(ROOT/'binding.txt').read_text(),cuda_sources=(ROOT/'cooperative_swizzle.cu').read_text(),
 functions=['attention'],extra_cuda_cflags=['-gencode=arch=compute_90a,code=sm_90a','-O3','--use_fast_math','-lineinfo','--ptxas-options=-v'],extra_cflags=['-O3'],verbose=True)

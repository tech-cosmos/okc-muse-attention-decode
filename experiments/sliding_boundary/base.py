from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline
ROOT=Path(__file__).resolve().parent
BUILD=ROOT/'.build-base';BUILD.mkdir(exist_ok=True)
module=load_inline(name='codex_sliding_base_rows',build_directory=str(BUILD),
 cpp_sources=(ROOT/'binding.txt').read_text(),cuda_sources=(ROOT/'base.cu').read_text(),
 functions=['attention'],extra_cuda_cflags=['-gencode=arch=compute_90a,code=sm_90a','-O3','--use_fast_math','-lineinfo','--ptxas-options=-v'],extra_cflags=['-O3'],verbose=True)

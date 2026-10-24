from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline
ROOT=Path(__file__).resolve().parent
ext=load_inline(name='muse_verify_merge22',cpp_sources='#include <torch/extension.h>\nvoid merge(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int,int,int,int,bool,bool);',cuda_sources=(ROOT/'native.cu').read_text(),functions=['merge'],extra_cuda_cflags=['-gencode=arch=compute_90a,code=sm_90a','-O3','--use_fast_math','--ptxas-options=-v'],extra_cflags=['-O3'],verbose=True)

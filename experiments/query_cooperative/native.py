"""Isolated native binding for sanitizer runs; no Triton loader kernels."""
from pathlib import Path
import os
from torch.utils.cpp_extension import load_inline
ROOT=Path(__file__).resolve().parent
variant=os.environ.get('QUERY_VARIANT','candidate')
assert variant in ('candidate','swizzled','distributed')
name={'candidate':'muse_native_query_cooperative_s33','swizzled':'muse_native_query_cooperative_swizzle_s33','distributed':'muse_native_query_cooperative_distributed_s33'}[variant]
module=load_inline(name=name,
    cpp_sources=(ROOT/'binding.cpp').read_text(),
    cuda_sources=(ROOT/(variant+'.cu')).read_text(),functions=['attention'],
    extra_cuda_cflags=['-gencode=arch=compute_90a,code=sm_90a','-O3','--use_fast_math'],
    extra_cflags=['-O3'],verbose=False)

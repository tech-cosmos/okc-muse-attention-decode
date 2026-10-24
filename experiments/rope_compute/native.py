"""Isolated native binding for sanitizer and arithmetic accuracy tests."""
from pathlib import Path
import os
from torch.utils.cpp_extension import load_inline
ROOT=Path(__file__).resolve().parent
variant=os.environ.get('ROPE_VARIANT','candidate')
assert variant in ('candidate','compensated')
name='muse_native_rope_compute_s34' if variant=='candidate' else 'muse_native_rope_compensated_s34'
module=load_inline(name=name,
    cpp_sources=(ROOT/'binding.cpp').read_text(),cuda_sources=(ROOT/(variant+'.cu')).read_text(),
    functions=['attention','rope_probe'],
    extra_cuda_cflags=['-gencode=arch=compute_90a,code=sm_90a','-O3','--use_fast_math'],
    extra_cflags=['-O3'],verbose=False)

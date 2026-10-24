from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline
ROOT=Path(__file__).resolve().parent
BUILD=ROOT/'.build-pruned';BUILD.mkdir(exist_ok=True)
module=load_inline(name='codex_sliding_pruned_rows',build_directory=str(BUILD),
 cpp_sources=(ROOT/'binding.txt').read_text(),cuda_sources=(ROOT/'pruned.cu').read_text(),
 functions=['attention'],extra_cuda_cflags=['-gencode=arch=compute_90a,code=sm_90a','-O3','--use_fast_math','-lineinfo','--ptxas-options=-v'],extra_cflags=['-O3'],verbose=True)

def native(data):
 # Independent globals keep the production module and timing control unchanged.
 import types
 from experiments.sliding_boundary import candidate
 import submission
 ns=dict(candidate.__dict__)
 ns['parent']=types.SimpleNamespace(**{**submission.__dict__,'_NATIVE_SLIDING':module})
 return types.FunctionType(candidate.native.__code__,ns)(data)

def kernel_fn(data):
 import submission
 from experiments.sliding_boundary import candidate
 c=data['config']
 if candidate.eligible(data) and c['batch']*c['num_kv_heads']<=4:return native(data)
 return submission.kernel_fn(data)

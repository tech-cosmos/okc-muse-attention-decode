from pathlib import Path
import ast
m=ast.parse(Path('experiments/sliding/cuda_merge_warp.py').read_text());const={n.targets[0].id:ast.literal_eval(n.value) for n in m.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('CPP','CUDA')}
cpp=const['CPP'].replace(',int warps,bool pdl',',bool pdl')
cu=const['CUDA'].replace(',int warps,bool pdl',',bool pdl')
cu=cu.replace('int b=qkv.size(0);','int b=qkv.size(0);constexpr int warps=1;')
a=cu.index(' #define WW(');z=cu.index(' C10_CUDA_KERNEL_LAUNCH_CHECK();',a)
cu=cu[:a]+' PP(32,1);\n'+cu[z:]
cu=cu.replace(' if(PDL)asm volatile',' // Partial reads must stay after this wait. Volatile PTX loads below prevent nvcc\n // from hoisting read-only loads across the dependency barrier.\n if(PDL)asm volatile')
s=Path('submission.py').read_text();a=s.index('_NATIVE_MERGE = load_inline(');z=s.index('\n\nRMS_EPS',a)
s=s[:a]+'''_NATIVE_MERGE = load_inline(
    name="muse_native_merge",
    cpp_sources=r\'\'\'
'''+cpp+'''\'\'\',
    cuda_sources=r\'\'\'
'''+cu+'''\'\'\',
    functions=["combine"],
    extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3", "--use_fast_math"],
    extra_cflags=["-O3"],
    verbose=False,
)
'''+s[z:]
s=s.replace('# Small sliding reductions benefit from batching four query heads in one CUDA CTA.\n    # Larger verify grids retain the existing parallel reduction.\n    if cfg["window"] and splits in (4, 8, 16, 32) and (length == 1 or batch * length * hq <= 2048):','# A warp cooperates across the 32 splits and emits one query head. This helps\n    # small batches; larger grids use main\'s batched Triton reduction.\n    if cfg["window"] and splits == 32 and batch * hkv <= 4:')
Path('submission.py').write_text(s)

from pathlib import Path
import ast
p=Path('experiments/sliding/cuda_merge.py')
m=ast.parse(p.read_text());const={n.targets[0].id:ast.literal_eval(n.value) for n in m.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('CPP','CUDA')}
cpp=const['CPP'].replace(',int warps,bool pdl',',bool pdl')
cu=const['CUDA'].replace(',int warps,bool pdl',',bool pdl')
cu=cu.replace('int b=qkv.size(0);','int b=qkv.size(0);constexpr int warps=4;')
cu=cu.replace(' #define WW(S) if(warps==1){PP(S,1);}else if(warps==2){PP(S,2);}else if(warps==4){PP(S,4);}else{PP(S,8);}',' #define WW(S) PP(S,4)')
cu=cu.replace('case 12:{WW(12);break;}','').replace('case 24:{WW(24);break;}','').replace('case 64:{WW(64);break;}','').replace('case 128:{WW(128);break;}','')
s=Path('experiments/sliding/submission_parent.py').read_text()
a=s.index('\nRMS_EPS =')
s=s[:a]+'''\n# A native warp merges one head's FP32 partials, then normalizes and gates the output.
# Build once at import; input-derived data is read only inside the timed kernels.
from torch.utils.cpp_extension import load_inline

_NATIVE_MERGE = load_inline(
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

'''+s[a:]
needle='''    _combine[(batch * length, hq)](
'''
s=s.replace(needle,'''    # Small sliding reductions benefit from batching four query heads in one CUDA CTA.
    # Larger verify grids retain the existing parallel reduction.
    if cfg["window"] and splits in (4, 8, 16, 32) and (length == 1 or batch * length * hq <= 2048):
        _NATIVE_MERGE.combine(qkv, pacc, pml, data["v_scale"], out,
                              hq, hkv, length, splits, USE_PDL)
        return out
'''+needle)
s=s.replace('Gated attention decode in two Triton launches.','Gated attention decode with Triton attention and a native CUDA merge for small sliding reductions.')
Path('submission.py').write_text(s)

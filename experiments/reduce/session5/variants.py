"""Native reduction variants; production source/plans remain unchanged."""
from pathlib import Path
import ast
from torch.utils.cpp_extension import load_inline
node=next(n.value for n in ast.parse(Path('submission.py').read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_NATIVE_MERGE' for t in n.targets))
base=next(ast.literal_eval(k.value) for k in node.keywords if k.arg=='cuda_sources')
cpp=next(ast.literal_eval(k.value) for k in node.keywords if k.arg=='cpp_sources')
a=base.index(' float2 a[N];');b=base.index(' float x=0,y=0;',a)
loads=r'''
 float2 a[N];
 const float* src=pa+((size_t)base+(size_t)warp*N*MG)*64+2*lane;
 size_t stride=(size_t)MG*64;
 #pragma unroll
 for(int i=0;i<N;i++) {
   int s=warp*N+i;
   a[i]=s<S?read2(src+i*stride):make_float2(0.f,0.f);
 }
'''
versions={'address':base[:a]+loads+base[b:]}
loads2=r'''
 float2 a[N];
 const float* src=pa+((size_t)base+(size_t)warp*N*MG)*64+2*lane;
 size_t stride=(size_t)MG*64;
 if((warp+1)*N<=S) {
   #pragma unroll
   for(int i=0;i<N;i++)a[i]=read2(src+i*stride);
 } else {
   #pragma unroll
   for(int i=0;i<N;i++)a[i]=warp*N+i<S?read2(src+i*stride):make_float2(0.f,0.f);
 }
'''
versions['unmasked']=base[:a]+loads2+base[b:]
# Include the cooperative final weights as a separate baseline follow-up.
import sys
sys.path.insert(0,'experiments/reduce')
import cooperative
versions['cooperative']=cooperative.CUDA
mods={'cooperative':cooperative.MOD}
for name in ['address','unmasked']:
    source=versions[name]
    Path('experiments/reduce/session5/'+name+'.cu').write_text(source)
    mods[name]=load_inline(name='muse_reduce_s5_'+name,cpp_sources=cpp,cuda_sources=source,functions=['combine_full'],extra_cuda_cflags=['-gencode=arch=compute_90a,code=sm_90a','-O3','--use_fast_math','-lineinfo'],extra_cflags=['-O3'],verbose=False)

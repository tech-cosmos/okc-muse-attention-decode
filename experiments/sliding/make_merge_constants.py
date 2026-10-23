from pathlib import Path
import ast
s=Path('submission.py').read_text();tree=ast.parse(s)
call=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_NATIVE_MERGE' for t in n.targets))
kwargs={k.arg:ast.literal_eval(k.value) for k in call.keywords}
cu=kwargs['cuda_sources']
cu=cu.replace('template<int S,int W,bool PDL>','template<int S,int W,bool PDL,int HQ,int L>')
cu=cu.replace(' int HQ,int HKV,int L,int B) {',' int B) {\n constexpr int HKV=HQ/8;')
cu=cu.replace('#define RUN(S,W,P)', '#define RUN(S,W,P,H,L)')
cu=cu.replace('combine_kernel<S,W,P>', 'combine_kernel<S,W,P,H,L>')
cu=cu.replace('out.data_ptr(),hq,hkv,length,b)', 'out.data_ptr(),b)')
cu=cu.replace('#define PP(S,W) if(pdl){RUN(S,W,true);}else{RUN(S,W,false);}', '#define LL(S,W,P,H) if(length==1){RUN(S,W,P,H,1);}else{RUN(S,W,P,H,8);}\n #define HH(S,W,P) if(hq==16){LL(S,W,P,16);}else{LL(S,W,P,128);}\n #define PP(S,W) if(pdl){HH(S,W,true);}else{HH(S,W,false);}')
out='from pathlib import Path\nfrom torch.utils.cpp_extension import load_inline\nCPP='+repr(kwargs['cpp_sources'])+'\nCUDA='+repr(cu)+'''\n_build=Path(__file__).resolve().parent/'.build_merge_constants';_build.mkdir(exist_ok=True)
module=load_inline(name='codex_merge_constants',build_directory=str(_build),cpp_sources=CPP,cuda_sources=CUDA,functions=['combine'],extra_cuda_cflags=['-gencode=arch=compute_90a,code=sm_90a','-O3','--use_fast_math','-lineinfo','--ptxas-options=-v'],extra_cflags=['-O3'],verbose=True)
'''
Path('experiments/sliding/cuda_merge_constants.py').write_text(out)
Path('experiments/sliding/merge_constants.cu').write_text(cu)

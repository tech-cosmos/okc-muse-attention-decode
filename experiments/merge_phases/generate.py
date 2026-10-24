"""Offline instrumentation only; production source is never edited."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
s=(ROOT/'previous.py').read_text()
a=s.index('_NATIVE_MERGE = load_inline(');b=s.index('\n\nRMS_EPS',a)
part=s[a:b]
part=part.replace('name="muse_native_merge"','name="muse_native_merge_phases_s39"')
part=part.replace('int splits,bool pdl);','int splits,bool pdl,torch::Tensor trace,int phase);')
part=part.replace('template<int W,int N,bool PDL>',r'''template<int P,int I> __device__ __forceinline__ void mark(unsigned long long &a,unsigned long long &b) {
 if constexpr ((P>=1 && P<=7 && I==P-1) || (P==8 && I==0) || (P==9 && I==2))
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(a) :: "memory");
 if constexpr ((P>=1 && P<=7 && I==P) || (P==8 && I==7) || (P==9 && I==2))
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(b) :: "memory");
}
template<int P> __device__ __forceinline__ void save(unsigned long long* trace,int row,int warp,int lane,unsigned long long a,unsigned long long b) {
 if constexpr(P!=0) if(lane==0){trace[(row*8+warp)*2]=a;trace[(row*8+warp)*2+1]=b;}
}
template<int W,int N,bool PDL,int PHASE>''')
part=part.replace('int HQ,int HKV,int L,int B,int S) {','int HQ,int HKV,int L,int B,int S,unsigned long long* trace) {\n unsigned long long t0=0,t1=0;mark<PHASE,0>(t0,t1);')
part=part.replace(' if(PDL)asm volatile', ' mark<PHASE,1>(t0,t1);\n if(PDL)asm volatile')
part=part.replace(' constexpr int NS=', ' mark<PHASE,2>(t0,t1);\n constexpr int NS=')
part=part.replace(' float2 a[N];',' mark<PHASE,3>(t0,t1);\n float2 a[N];')
part=part.replace(' __shared__ float partial',' mark<PHASE,4>(t0,t1);\n __shared__ float partial')
part=part.replace('   if(warp!=0)return;','   mark<PHASE,5>(t0,t1);\n   if(warp!=0){save<PHASE>(trace,row,warp,lane,t0,t1);return;}')
part=part.replace(' x=x/den*vscale;', ' mark<PHASE,6>(t0,t1);\n x=x/den*vscale;')
part=part.replace('=__floats2bfloat162_rn(x,y);','=__floats2bfloat162_rn(x,y);\n mark<PHASE,7>(t0,t1);save<PHASE>(trace,row,warp,lane,t0,t1);')
part=part.replace('int splits,bool pdl) {','int splits,bool pdl,at::Tensor trace,int phase) {')
part=part.replace('#define FULL_RUN(P)', '#define FULL_RUN(P,T)')
part=part.replace('reduce_kernel<8,16,P>', 'reduce_kernel<8,16,P,T>')
part=part.replace('hq,hkv,length,b,splits))','hq,hkv,length,b,splits,(unsigned long long*)trace.data_ptr()))')
part=part.replace(' if(pdl){FULL_RUN(true);}else{FULL_RUN(false);}', ' switch(phase){\n'+''.join(f' case {p}: if(pdl){{FULL_RUN(true,{p});}}else{{FULL_RUN(false,{p});}}break;\n' for p in range(10))+' default: TORCH_CHECK(false,"invalid phase");\n }')
s=s[:a]+part+s[b:]
s+='''\n# Diagnostic wrapper retains trace storage only for the external offline reader.
_NATIVE_MERGE_IMPL = _NATIVE_MERGE
TRACE = None
PHASE = 8
class _MergeProbe:
    @staticmethod
    def combine_full(qkv,pa,pm,vs,out,hq,hkv,length,splits,pdl):
        global TRACE
        TRACE=torch.empty((qkv.shape[0]*length*hq,8,2),device=qkv.device,dtype=torch.int64)
        _NATIVE_MERGE_IMPL.combine_full(qkv,pa,pm,vs,out,hq,hkv,length,splits,pdl,TRACE,PHASE)
_NATIVE_MERGE=_MergeProbe()
'''
(ROOT/'candidate.py').write_text(s)
print('generated',len(s))

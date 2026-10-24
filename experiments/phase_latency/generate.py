"""Instrument one adjacent phase interval per compiled kernel; no algorithm edits."""
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
source=(ROOT/'production_snapshot.py').read_text()
tree=ast.parse(source)
node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_NATIVE_SLIDING' for t in n.targets))
kw={k.arg:ast.literal_eval(k.value) for k in node.value.keywords}
cu=kw['cuda_sources'];cpp=kw['cpp_sources']
# Compile controls/individual intervals from the same original kernel.
cu=cu.replace('template<int PAGE,bool EARLY>', 'template<int PAGE,bool EARLY,int PHASE>')
cu=cu.replace('int MP,int S,float scale) {','int MP,int S,float scale,unsigned long long* trace) {')
cu=cu.replace('const int b=blockIdx.y,h=blockIdx.z,split=blockIdx.x;', 'const int b=blockIdx.y,h=blockIdx.z,split=blockIdx.x;\n unsigned long long tbegin=0,tend=0;\n MARK0')
anchors=[
(' if(EARLY)gather();','after'),
(' asm volatile("cp.async.wait_group 0;":::"memory");','before'),
(' asm volatile("cp.async.wait_group 0;":::"memory");','after'),
(' __syncthreads();\n if(own && warp==1) {','between'),
(' __syncthreads();\n #pragma unroll\n for(int j=0;j<4;j++) {','between'),
(' __syncthreads();\n float score[32]={0};','between'),
(' float mx[2]={-1e30f,-1e30f},sum[2]={0,0};','before'),
(' float acc[32]={0};','before'),
(' if(warp==0) {\n  long outbase=','before'),
]
for i,(anchor,where) in enumerate(anchors,1):
    assert cu.count(anchor)==1,(i,anchor,cu.count(anchor))
    if where=='after':replacement=anchor+f'\n MARK{i}'
    elif where=='before':replacement=f' MARK{i}\n'+anchor
    else:replacement=anchor.replace(' __syncthreads();',' __syncthreads();'+f'\n MARK{i}')
    cu=cu.replace(anchor,replacement)
a='\n}\nvoid attention('
assert cu.count(a)==1
cu=cu.replace(a,'\n MARK10\n if constexpr(PHASE>=0) if(lane==0) {\n  long at=(((long)b*HKV+h)*S+split)*8+warp*2;\n  trace[at]=tbegin;trace[at+1]=tend;\n }'+a)
for i in reversed(range(11)):
    start=f'if constexpr(PHASE=={i}'+(' || PHASE==10 || PHASE==11' if i==0 else '')+') asm volatile("mov.u64 %0, %%clock64;" : "=l"(tbegin) :: "memory");'
    end=f'if constexpr(PHASE=={i-1}'+(' || PHASE==10' if i==10 else '')+') asm volatile("mov.u64 %0, %%clock64;" : "=l"(tend) :: "memory");' if i else 'if constexpr(PHASE==11) asm volatile("mov.u64 %0, %%clock64;" : "=l"(tend) :: "memory");'
    # PHASE=10 is total; there is no phase starting at the final boundary.
    if i==10:start=''
    cu=cu.replace(f'MARK{i}',start+'\n '+end)
cu=cu.replace('int page,double scale,bool early) {','int page,double scale,bool early,at::Tensor trace,int phase) {')
cpp=cpp.replace('int page,double scale,bool early);','int page,double scale,bool early,torch::Tensor trace,int phase);')
cu=cu.replace('#define RUN(P,E) sliding<P,E>', '#define RUN(P,F) sliding<P,true,F>')
cu=cu.replace('pt.size(1),S,scale)', 'pt.size(1),S,scale,(unsigned long long*)trace.data_ptr<int64_t>())')
a=' if(page==1){if(early){RUN(1,true);}else{RUN(1,false);}}else{if(early){RUN(64,true);}else{RUN(64,false);}}'
assert cu.count(a)==1
switch=' switch(phase) {\n'+''.join(f' case {p}: if(page==1){{RUN(1,{p});}}else{{RUN(64,{p});}}break;\n' for p in range(-1,12))+' default: TORCH_CHECK(false,"invalid diagnostic phase");\n }'
cu=cu.replace(a,switch)
(ROOT/'diagnostic.cu').write_text(cu)
(ROOT/'binding.cpp').write_text(cpp)
(ROOT/'results/source-manifest.json').write_text(json.dumps({f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['production_snapshot.py','diagnostic.cu','binding.cpp']},indent=2)+'\n')

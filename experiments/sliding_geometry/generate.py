"""Specialize native sliding strides for the two supported head geometries.

Retain the generic native instantiation for other geometries. Only the embedded
native sliding source changes; dispatch, arithmetic precision and work stay fixed.
"""
from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parent

def generate(s):
    begin = s.index('_NATIVE_SLIDING = load_inline(')
    end = s.index('\n_ROPE =', begin)
    native = s[begin:end]
    native = native.replace('muse_native_sliding_direct_v1', 'muse_native_sliding_geometry_s44')
    native = native.replace('template<int PAGE,bool EARLY>', 'template<int PAGE,bool EARLY,int FIXED_HKV>')
    needle = 'const float* ks,const float* vs,float* pa,float* pm,int HQ,int HKV,int MP,int S,float scale) {'
    assert native.count(needle) == 1
    native = native.replace(needle, '''const float* ks,const float* vs,float* pa,float* pm,int HQ_arg,int HKV_arg,int MP,int S,float scale) {
 const int HKV=FIXED_HKV ? FIXED_HKV : HKV_arg;
 const int HQ=FIXED_HKV ? FIXED_HKV*8 : HQ_arg;''')
    native = native.replace('#define RUN(P,E) sliding<P,E>', '#define RUN(P,E,H) sliding<P,E,H>')
    needle = ' if(page==1){if(early){RUN(1,true);}else{RUN(1,false);}}else{if(early){RUN(64,true);}else{RUN(64,false);}}'
    assert native.count(needle) == 1
    native = native.replace(needle, ''' #define HEAD_RUN(H) if(page==1){if(early){RUN(1,true,H);}else{RUN(1,false,H);}}else{if(early){RUN(64,true,H);}else{RUN(64,false,H);}}
 if(hkv==2 && hq==16){HEAD_RUN(2);}
 else if(hkv==16 && hq==128){HEAD_RUN(16);}
 else {HEAD_RUN(0);}
 #undef HEAD_RUN
 #undef RUN''')
    return s[:begin]+native+s[end:]

if __name__ == '__main__':
    previous = (ROOT/'previous.py').read_text()
    candidate = generate(previous)
    (ROOT/'candidate.py').write_text(candidate)
    for label, source in [('previous', previous), ('candidate', candidate)]:
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=='_NATIVE_SLIDING' for t in n.targets))
        keywords = {k.arg:ast.literal_eval(k.value) for k in node.value.keywords}
        (ROOT/f'{label}.cu').write_text(keywords['cuda_sources'])
    unsigned = candidate.replace('muse_native_sliding_geometry_s44', 'muse_native_sliding_geometry_unsigned_s44')
    unsigned = unsigned.replace('const int tid=threadIdx.x,warp=tid/32,lane=tid%32;', 'const unsigned tid=threadIdx.x,warp=tid/32,lane=tid%32;')
    unsigned = unsigned.replace('int i=tid+j*128,row=i/4,piece=(i%4)*16,t=lo+row;', 'unsigned i=tid+j*128,row=i/4,piece=(i%4)*16;int t=lo+row;')
    (ROOT/'unsigned.py').write_text(unsigned)
    node = next(n for n in ast.parse(unsigned).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_NATIVE_SLIDING' for t in n.targets))
    (ROOT/'unsigned.cu').write_text(next(ast.literal_eval(k.value) for k in node.value.keywords if k.arg=='cuda_sources'))
    (ROOT/'.gitignore').write_text('__pycache__/\n*.so\n')

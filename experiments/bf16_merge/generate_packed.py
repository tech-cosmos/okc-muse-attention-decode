"""Reproduce the final packed-lifetime experiment from the frozen bits variant."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
s=(ROOT/'bits.py').read_text().replace('muse_native_merge_bf16_bits','muse_native_merge_bf16_packed')
s=s.replace('__device__ __forceinline__ float2 read2(const __nv_bfloat16* p)', '__device__ __forceinline__ unsigned read_partial(const __nv_bfloat16* p)')
s=s.replace(' return make_float2(__uint_as_float(v << 16), __uint_as_float(v & 0xffff0000u));\n}', ''' return v;
}
__device__ __forceinline__ float2 read_partial(const float* p) { return read2(p); }
__device__ __forceinline__ float2 widen(float2 v) { return v; }
__device__ __forceinline__ float2 widen(unsigned v) {
 return make_float2(__uint_as_float(v << 16), __uint_as_float(v & 0xffff0000u));
}''')
s=s.replace(' float2 a[N];',' using Pair=decltype(read_partial(pa));\n Pair a[N];')
s=s.replace('a[i]=s<S?read2(pa+(base+s*MG)*64+2*lane):make_float2(0.f,0.f);','a[i]=s<S?read_partial(pa+(base+s*MG)*64+2*lane):Pair{};')
s=s.replace('x=fmaf(w,a[i].x,x);y=fmaf(w,a[i].y,y);','float2 v=widen(a[i]);\n   x=fmaf(w,v.x,x);y=fmaf(w,v.y,y);')
if __name__=='__main__':
    assert s==(ROOT/'candidate.py').read_text()
    print('Packed-lifetime candidate reproduced exactly')

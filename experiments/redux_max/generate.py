"""Replace only full-merge max32 with a bit-exact finite-FP32 ordered REDUX."""
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OLD='''__device__ __forceinline__ float max32(float x) {
 #pragma unroll
 for(int d=16;d;d/=2)x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,d));
 return x;
}'''
NEW='''__device__ __forceinline__ float max32(float x) {
 // Signed integer order agrees with finite float order after flipping the
 // magnitude bits of negative values. This is a reversible bit permutation.
 int bits=__float_as_int(x);
 int ordered=bits^((bits>>31)&0x7fffffff);
 int winner=__reduce_max_sync(0xffffffff,ordered);
 return __int_as_float(winner^((winner>>31)&0x7fffffff));
}'''
def generate():
 s=(ROOT/'previous.py').read_text();assert s.count(OLD)==1
 return s.replace('name="muse_native_merge"','name="muse_native_merge_redux_s40"').replace(OLD,NEW)
if __name__=='__main__':(ROOT/'candidate.py').write_text(generate())

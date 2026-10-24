"""Remove the new-value raw-tile dependency, preserving its FP8 round trip."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
base = (ROOT / 'base.cu').read_text()
barrier = ''' }
 __syncthreads();
 #pragma unroll
 for(int j=0;j<4;j++) {'''
assert base.count(barrier) == 1
# Control: only the block containing the new token needs the second barrier.
conditional = base.replace(barrier, barrier.replace('__syncthreads();', 'if(own)__syncthreads();'))
direct = base.replace(barrier, barrier.replace(' __syncthreads();\n', ''))
old = '  Vraw[row*64+lane]=v0;Vraw[row*64+lane+32]=v1;'
new = '''  // Quantize exactly as the cache store, then expand those bytes to FP16.
  __half2_raw vh=__nv_cvt_fp8x2_to_halfraw2((unsigned short)v0|((unsigned short)v1<<8),__NV_E4M3);
  unsigned i0=row*64+lane,i1=i0+32;
  ((unsigned short*)V)[i0 ^ (((i0>>6)&7)<<3)]=vh.x;
  ((unsigned short*)V)[i1 ^ (((i1>>6)&7)<<3)]=vh.y;'''
assert direct.count(old) == 1
direct = direct.replace(old, new)
old = '''  unsigned long long bytes=*(unsigned long long*)(Vraw+i);unsigned cv[4];'''
assert direct.count(old) == 1
direct = direct.replace(old, '''  // Other threads must not read or overwrite the directly published row.
  if(own && (int)(i/64)==seq-lo)continue;
''' + old)
wrapper = (ROOT / 'base.py').read_text()
for name, source in [('conditional', conditional), ('direct', direct)]:
    (ROOT / f'{name}.cu').write_text(source)
    (ROOT / f'{name}.py').write_text(wrapper.replace('base', name))

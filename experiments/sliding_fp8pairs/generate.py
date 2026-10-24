"""Pair FP32->E4M3 conversions without changing rounding or attention."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
base = (ROOT / 'base.cu').read_text()
old = 'packed|=(unsigned)to8(a)<<(8*k);packed|=(unsigned)to8(bb)<<(8*(k+1));'
new = 'packed|=(unsigned)__nv_cvt_float2_to_fp8x2(make_float2(a,bb),__NV_SATFINITE,__NV_E4M3)<<(8*k);'
assert base.count(old) == 1
query = base.replace(old, new)
old_new = 'int row=seq-lo;unsigned char k0=to8(nx),k1=to8(ny),v0=to8(nv0),v1=to8(nv1);'
new_new = '''int row=seq-lo;
  unsigned kp=__nv_cvt_float2_to_fp8x2(make_float2(nx,ny),__NV_SATFINITE,__NV_E4M3);
  unsigned vp=__nv_cvt_float2_to_fp8x2(make_float2(nv0,nv1),__NV_SATFINITE,__NV_E4M3);
  unsigned char k0=kp,k1=kp>>8,v0=vp,v1=vp>>8;'''
assert query.count(old_new) == 1
both = query.replace(old_new, new_new)
wrapper = (ROOT / 'base.py').read_text()
for name, source in [('query', query), ('both', both)]:
    (ROOT / f'{name}.cu').write_text(source)
    (ROOT / f'{name}.py').write_text(wrapper.replace('base', name))

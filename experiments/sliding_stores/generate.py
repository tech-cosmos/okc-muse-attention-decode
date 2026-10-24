from pathlib import Path

ROOT = Path(__file__).resolve().parent
base = (ROOT / 'base.cu').read_text()
old = '''  for(int i=0;i<32;i++)if(!(i&2)) {
   int row=lane/4,col=(i/8)*16+(lane%4)*2+(i&1)+(i&4)*2;
   pa[(outbase+row)*64+col]=acc[i];
  }
  if(lane%4==0){pm[(outbase+lane/4)*2]=mx[0];pm[(outbase+lane/4)*2+1]=sum[0];}'''
assert base.count(old) == 1
vector = '''  for(int i=0;i<32;i+=2)if(!(i&2)) {
   int row=lane/4,col=(i/8)*16+(lane%4)*2+(i&4)*2;
   ((float2*)(pa+(outbase+row)*64+col))[0]=make_float2(acc[i],acc[i+1]);
  }
  if(lane%4==0){((float2*)pm)[outbase+lane/4]=make_float2(mx[0],sum[0]);}'''
explicit = vector.replace('((float2*)(pa+(outbase+row)*64+col))[0]=make_float2(acc[i],acc[i+1]);',
 '''asm volatile("st.global.v2.f32 [%0],{%1,%2};"::"l"(pa+(outbase+row)*64+col),"f"(acc[i]),"f"(acc[i+1]):"memory");''').replace(
 '((float2*)pm)[outbase+lane/4]=make_float2(mx[0],sum[0]);',
 '''asm volatile("st.global.v2.f32 [%0],{%1,%2};"::"l"(pm+(outbase+lane/4)*2),"f"(mx[0]),"f"(sum[0]):"memory");''')
loader = (ROOT / 'base.py').read_text()
for name, epilogue in [('vector', vector), ('explicit', explicit)]:
    (ROOT / (name+'.cu')).write_text(base.replace(old, epilogue))
    (ROOT / (name+'.py')).write_text(loader.replace('base', name))

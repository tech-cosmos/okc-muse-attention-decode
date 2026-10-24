"""Move independent prologue loads ahead of the gathered-page dependency."""
import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parent
mod=ast.parse(Path('submission.py').read_text())
call=next(n.value for n in mod.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_NATIVE_SLIDING' for t in n.targets))
base=ast.literal_eval(next(k.value for k in call.keywords if k.arg=='cuda_sources'))
assert base==(ROOT/'base.cu').read_text()
old='''  float x[16],sq=0;
  #pragma unroll
  for(int i=0;i<16;i++) {
   int col=(i/8)*32+(lane%4)*4+((i%8)/4)*16+i%4;
   x[i]=__bfloat162float(qkv[b*width+(h*8+lane/4)*64+col]);sq+=x[i]*x[i];
  }
'''
new='''  float x[16],sq=0;
  #pragma unroll
  for(int i=0;i<16;i++) {
   x[i]=__bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&qx[i]));sq+=x[i]*x[i];
  }
'''
pre=''' // Independent Q bytes remain unconsumed until after page-dependent copies.
 unsigned short qx[16];
 if(warp==0) {
  #pragma unroll
  for(int i=0;i<16;i++) {
   int col=(i/8)*32+(lane%4)*4+((i%8)/4)*16+i%4;
   const __nv_bfloat16* qp=qkv+b*width+(h*8+lane/4)*64+col;
   asm volatile("ld.global.b16 %0,[%1];":"=h"(qx[i]):"l"(qp):"memory");
  }
 }
'''
assert old in base
qpre=base.replace(' if(EARLY)gather();',pre+' if(EARLY)gather();',1).replace(old,new,1)
# Same arithmetic, compiler-visible float loads as a scheduling control.
plainpre=pre.replace('unsigned short qx[16];','float qx[16];').replace('   asm volatile("ld.global.b16 %0,[%1];":"=h"(qx[i]):"l"(qp):"memory");','   qx[i]=__bfloat162float(*qp);')
plainnew=new.replace('__bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&qx[i]))','qx[i]')
plain=base.replace(' if(EARLY)gather();',plainpre+' if(EARLY)gather();',1).replace(old,plainnew,1)
# RoPE loads have the same sequence dependence as page addresses; preload all
# query-specific coefficients as raw scalar operands, then consume after gather.
ropepre=''' float qc[8],qs[8];
 if(warp==0) {
  #pragma unroll
  for(int j=0;j<8;j++) {
   int col=(j/4)*32+(lane%4)*4+((j%4)/2)*16+(j%2)*2;
   const float* p=cs+(long)seq*64+col/2;
   asm volatile("ld.global.f32 %0,[%2]; ld.global.f32 %1,[%3];":"=f"(qc[j]),"=f"(qs[j]):"l"(p),"l"(p+32):"memory");
  }
 }
'''
rope=qpre.replace(' if(EARLY)gather();',ropepre+' if(EARLY)gather();',1).replace('float cc=cs[(long)seq*64+col/2],ss=cs[(long)seq*64+col/2+32];','float cc=qc[ix/2],ss=qs[ix/2];')
# Volatile query and page accesses preserve their issue order at PTX level;
# unlike an operand fence this does not consume query data before page loads.
ordered=qpre.replace('ld.global.b16','ld.volatile.global.b16').replace('int page=valid?pt[b*MP+t/PAGE]:0;', 'int page=valid?*reinterpret_cast<const volatile int*>(pt+b*MP+t/PAGE):0;')
for name,src in [('qpre',qpre),('plain',plain),('rope',rope),('ordered',ordered)]:
 (ROOT/(name+'.cu')).write_text(src)
 (ROOT/(name+'.py')).write_text((ROOT/'base.py').read_text().replace('.build-base','.build-'+name).replace('codex_sliding_loads_base','codex_sliding_loads_'+name).replace("'base.cu'",repr(name+'.cu')))

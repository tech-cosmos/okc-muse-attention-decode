"""Distribute eight real query heads over all 128 existing CTA threads.

Only 512 shared bytes are handed back to warp zero, at an existing barrier.
There are no shared padded query rows and no additional launch/barrier.
"""
from pathlib import Path
import ast, hashlib, json
ROOT = Path(__file__).resolve().parent

def generate(s):
    a = s.index(' unsigned qr[2][4]={{0}};')
    b = s.index(' // Keep new K/V', a)
    prep = ''' // Two real query heads per warp, sixteen lanes per head.
 const int qhead=warp*2+lane/16,qcol=(lane%16)*4;
 float x[4],sq=0;
 #pragma unroll
 for(int k=0;k<4;k++) {
  x[k]=__bfloat162float(qkv[b*width+(h*8+qhead)*64+qcol+k]);
  sq+=x[k]*x[k];
 }
 #pragma unroll
 for(int d=8;d;d/=2)sq+=__shfl_xor_sync(0xffffffff,sq,d,16);
 float qinv=rsqrtf(sq/64.f+1e-5f);
 unsigned packed=0;
 #pragma unroll
 for(int k=0;k<4;k+=2) {
  float cc=cs[(long)seq*64+(qcol+k)/2],ss=cs[(long)seq*64+(qcol+k)/2+32];
  float a=(x[k]*cc-x[k+1]*ss)*qinv,bb=(x[k+1]*cc+x[k]*ss)*qinv;
  packed|=(unsigned)to8(a)<<(8*k);packed|=(unsigned)to8(bb)<<(8*(k+1));
 }
 Qpacked[qhead*16+lane%16]=packed;
'''
    s=s[:a]+prep+s[b:]
    s=s.replace(' __shared__ __align__(1024) half V[4096];',
                ' __shared__ __align__(1024) half V[4096];\n __shared__ unsigned Qpacked[8*16];')
    needle=' __syncthreads();\n if(own && warp==1) {'
    assert s.count(needle)==1
    s=s.replace(needle,''' __syncthreads();
 unsigned qr[2][4]={{0}};
 if(warp==0) {
  #pragma unroll
  for(int d=0;d<2;d++) {
   #pragma unroll
   for(int i=0;i<2;i++)qr[d][i*2]=Qpacked[(lane/4)*16+d*8+(lane%4)+i*4];
  }
 }
 if(own && warp==1) {''')
    return s.replace('muse_native_sliding_direct_v1','muse_native_query_cooperative_s33')

if __name__=='__main__':
    s=(ROOT/'production_snapshot.py').read_text()
    v=generate(s)
    (ROOT/'candidate.py').write_text(v)
    sw=v.replace('muse_native_query_cooperative_s33','muse_native_query_cooperative_swizzle_s33')
    sw=sw.replace('Qpacked[qhead*16+lane%16]', 'Qpacked[qhead*16+((lane%16)^((qhead/2)*4))]')
    sw=sw.replace('Qpacked[(lane/4)*16+d*8+(lane%4)+i*4]', 'Qpacked[(lane/4)*16+((d*8+(lane%4)+i*4)^((lane/8)*4))]')
    (ROOT/'swizzled.py').write_text(sw)
    # Avoid cross-warp handoff altogether by placing two real Q rows in each
    # warp's WGMMA row block. Warp shuffles map the sixteen-lane preparation
    # layout to the four-lane WGMMA fragment, then each warp writes two heads.
    dist=v.replace('muse_native_query_cooperative_s33','muse_native_query_cooperative_distributed_s33')
    dist=dist.replace(' __shared__ unsigned Qpacked[8*16];\n','')
    dist=dist.replace(' Qpacked[qhead*16+lane%16]=packed;',''' unsigned qr[2][4]={{0}};
 #pragma unroll
 for(int d=0;d<2;d++) {
  #pragma unroll
  for(int i=0;i<2;i++) {
   int src=((lane/4)*16+d*8+(lane%4)+i*4)&31;
   unsigned word=__shfl_sync(0xffffffff,packed,src);
   qr[d][i*2]=lane<8?word:0;
  }
 }''')
    handoff=''' unsigned qr[2][4]={{0}};
 if(warp==0) {
  #pragma unroll
  for(int d=0;d<2;d++) {
   #pragma unroll
   for(int i=0;i<2;i++)qr[d][i*2]=Qpacked[(lane/4)*16+d*8+(lane%4)+i*4];
  }
 }
'''
    assert dist.count(handoff)==1
    dist=dist.replace(handoff,'')
    old=''' if(warp==0) {
  long outbase=((long)(b*HKV+h)*S+split)*8;'''
    assert dist.count(old)==1
    dist=dist.replace(old,''' if(lane<8) {
  long outbase=((long)(b*HKV+h)*S+split)*8+warp*2;''')
    (ROOT/'distributed.py').write_text(dist)
    tree=ast.parse(v)
    for n in tree.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_NATIVE_SLIDING' for t in n.targets):
            for kw in n.value.keywords:
                if kw.arg in ('cpp_sources','cuda_sources'):
                    (ROOT/('binding.cpp' if kw.arg=='cpp_sources' else 'candidate.cu')).write_text(ast.literal_eval(kw.value))
    for name,text in [('swizzled',sw),('distributed',dist)]:
        tree=ast.parse(text)
        node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_NATIVE_SLIDING' for t in n.targets))
        (ROOT/(name+'.cu')).write_text(ast.literal_eval(next(k.value for k in node.value.keywords if k.arg=='cuda_sources')))
    (ROOT/'results/source-manifest.json').write_text(json.dumps({p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['production_snapshot.py','candidate.py','candidate.cu','swizzled.py','swizzled.cu','distributed.py','distributed.cu','binding.cpp']},indent=2)+'\n')

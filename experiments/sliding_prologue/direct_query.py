from pathlib import Path
import importlib.util
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('gen',ROOT/'generate_base.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
s=(ROOT/'kernel.cu').read_text()
s=s.replace('unsigned char Vraw[4096],Q[64*64];','unsigned char Vraw[4096];')
a=s.index(' #pragma unroll\n for(int row=warp;row<64;row+=4)')
z=s.index(' // Keep new K/V',a)
s=s[:a]+r'''
 unsigned qr[2][4]={{0}};
 if(warp==0) {
  float x[16],sq=0;
  #pragma unroll
  for(int i=0;i<16;i++) {
   int col=(i/8)*32+(lane%4)*4+((i%8)/4)*16+i%4;
   x[i]=__bfloat162float(qkv[b*width+(h*8+lane/4)*64+col]);sq+=x[i]*x[i];
  }
  sq+=__shfl_xor_sync(0xffffffff,sq,1);sq+=__shfl_xor_sync(0xffffffff,sq,2);
  float inv=rsqrtf(sq/64.f+1e-5f);
  #pragma unroll
  for(int d=0;d<2;d++) {
   #pragma unroll
   for(int i=0;i<2;i++) {
    unsigned packed=0;
    #pragma unroll
    for(int k=0;k<4;k+=2) {
     int col=d*32+(lane%4)*4+i*16+k,ix=d*8+i*4+k;
     float cc=cs[(long)seq*64+col/2],ss=cs[(long)seq*64+col/2+32];
     float a=(x[ix]*cc-x[ix+1]*ss)*inv,bb=(x[ix+1]*cc+x[ix]*ss)*inv;
     packed|=(unsigned)to8(a)<<(8*k);packed|=(unsigned)to8(bb)<<(8*(k+1));
    }
    qr[d][i*2]=packed;
   }
  }
 }
''' + s[z:]
s=s.replace('own && warp==0','own && warp==1')
a=s.index(' unsigned qr[2][4];');z=s.index(' float score[32]',a);s=s[:a]+s[z:]
(ROOT/'direct.cu').write_text(s)
p=(ROOT/'native.py').read_text().replace("ROOT/'.build'","ROOT/'.build-direct'").replace("codex_sliding_prologue'","codex_sliding_direct'").replace("ROOT/'kernel.cu'","ROOT/'direct.cu'")
(ROOT/'direct.py').write_text(p)

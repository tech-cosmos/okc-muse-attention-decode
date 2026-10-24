"""Build L8 single-tile attention from the exact session13 L1 source.
The source transformation keeps all MMA layouts and cache precision unchanged.
"""
import ast
from pathlib import Path
R=Path(__file__).resolve().parent
source=(R/'production_parent.py').read_text()
tree=ast.parse(source)
call=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_NATIVE_SLIDING' for t in n.targets))
cu=next(ast.literal_eval(k.value) for k in call.keywords if k.arg=='cuda_sources')
cu=cu.replace(' if(warp==0) {\n  float x[16],sq=0;', ' for(int r=0;r<2;r++) {\n  int token=warp*2+r;\n  float x[16],sq=0;')
cu=cu.replace('qkv[b*width+(h*8+lane/4)*64+col]', 'qkv[(b*8+token)*width+(h*8+lane/4)*64+col]')
cu=cu.replace('cs[(long)seq*64+col/2]', 'cs[(long)(seq+token)*64+col/2]').replace('cs[(long)seq*64+col/2+32]', 'cs[(long)(seq+token)*64+col/2+32]')
cu=cu.replace('qr[d][i*2]=packed;', 'qr[d][i*2+r]=packed;')
a=cu.index(' float c0=');b=cu.index('\n unsigned qr',a);cu=cu[:a]+cu[b:]
a=cu.index(' // Keep new K/V');b=cu.index(' __syncthreads();\n #pragma unroll\n for(int j=0;j<4;j++)',a)
cu=cu[:a]+''' // Each warp prepares up to two new tokens; only their natural tile writes them.
 float nx[2]={0,0},ny[2]={0,0},nv0[2]={0,0},nv1[2]={0,0};
 bool own[2];
 #pragma unroll
 for(int j=0;j<2;j++) {
  int token=warp+4*j,pos=seq+token;
  own[j]=pos>=lo && pos<lo+64;
  if(own[j]) {
   const __nv_bfloat16* p=qkv+(b*8+token)*width+(HQ+h)*64;
   nx[j]=__bfloat162float(p[lane]);ny[j]=__bfloat162float(p[lane+32]);
   float inv=rsqrtf(sum32(nx[j]*nx[j]+ny[j]*ny[j])/64.f+1e-5f)/kscale;
   float xp=__shfl_xor_sync(0xffffffff,nx[j],1),yp=__shfl_xor_sync(0xffffffff,ny[j],1),sign=lane%2?1.f:-1.f;
   float c0=cs[(long)pos*64+lane/2],s0=cs[(long)pos*64+lane/2+32];
   float c1=cs[(long)pos*64+(lane+32)/2],s1=cs[(long)pos*64+(lane+32)/2+32];
   nx[j]=(nx[j]*c0+sign*xp*s0)*inv;ny[j]=(ny[j]*c1+sign*yp*s1)*inv;
   nv0[j]=__bfloat162float(p[HKV*64+lane])/vscale;nv1[j]=__bfloat162float(p[HKV*64+lane+32])/vscale;
  }
 }
 if(!EARLY)gather();
 asm volatile("cp.async.wait_group 0;":::"memory");
 __syncthreads();
 #pragma unroll
 for(int j=0;j<2;j++)if(own[j]) {
  int token=warp+4*j,row=seq+token-lo;
  unsigned char k0=to8(nx[j]),k1=to8(ny[j]),v0=to8(nv0[j]),v1=to8(nv1[j]);
  K[sw64(row*64+lane)]=k0;K[sw64(row*64+lane+32)]=k1;
  Vraw[row*64+lane]=v0;Vraw[row*64+lane+32]=v1;
  long slot=loc[b*8+token]*HKV*64+h*64;
  kc[slot+lane]=k0;kc[slot+lane+32]=k1;vc[slot+lane]=v0;vc[slot+lane+32]=v1;
 }
''' +cu[b:]
cu=cu.replace('score[i]=lo+col<=seq?score[i]:-1e30f;', 'int pos=seq+warp*2+(i&2)/2;\n  score[i]=(lo+col<=pos && lo+col>=pos-2047)?score[i]:-1e30f;')
cu=cu.replace('float p=lo+col<=seq?exp2f', 'int pos=seq+warp*2+r;\n  float p=(lo+col<=pos && lo+col>=pos-2047)?exp2f')
a=cu.index(' if(warp==0) {\n  long outbase');b=cu.index('\n}\nvoid attention',a)
cu=cu[:a]+''' long outbase=((long)(b*HKV+h)*S+split)*64;
 #pragma unroll
 for(int i=0;i<32;i++) {
  int row=warp*16+lane/4+((i&2)/2)*8,col=(i/8)*16+(lane%4)*2+(i&1)+(i&4)*2;
  pa[(outbase+row)*64+col]=acc[i];
 }
 if(lane%4==0) {
  #pragma unroll
  for(int r=0;r<2;r++) {
   int row=warp*16+lane/4+r*8;
   pm[(outbase+row)*2]=mx[r];pm[(outbase+row)*2+1]=sum[r];
  }
 }
''' +cu[b:]
(R/'kernel.cu').write_text(cu)

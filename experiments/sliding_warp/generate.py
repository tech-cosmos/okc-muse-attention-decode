"""Replace the padded WGMMA arithmetic with a single warp's 16-row MMA tile."""
from pathlib import Path
P=Path(__file__).resolve().parent
base=(P/'base.cu').read_text()
s=base.replace('__launch_bounds__(128)','__launch_bounds__(THREADS)')
s=s.replace('for(int j=0;j<2;j++) {\n   int i=tid+j*128','for(int j=0;j<256/THREADS;j++) {\n   int i=tid+j*THREADS')
s=s.replace('own && warp==1','own && warp==(THREADS==32?0:1)')
s=s.replace('for(int j=0;j<4;j++) {\n  unsigned i=8*tid+j*1024,target=i ^ (((i>>6)&7)<<3);','for(int j=0;j<512/THREADS;j++) {\n  unsigned i=8*tid+j*(8*THREADS),target=i;')
a=s.index(' float score[32]={0};')
b=s.index(' float mx[2]',a)
s=s[:a]+r'''
 if(warp==0) {
 float score[32]={0};
 #pragma unroll
 for(int n=0;n<8;n++) {
  #pragma unroll
  for(int d=0;d<2;d++) {
   unsigned kr0=*(unsigned*)(K+sw64((n*8+lane/4)*64+d*32+(lane%4)*4));
   unsigned kr1=*(unsigned*)(K+sw64((n*8+lane/4)*64+d*32+(lane%4)*4+16));
   asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    : "+f"(score[n*4]),"+f"(score[n*4+1]),"+f"(score[n*4+2]),"+f"(score[n*4+3])
    : "r"(qr[d][0]),"r"(qr[d][1]),"r"(qr[d][2]),"r"(qr[d][3]),"r"(kr0),"r"(kr1));
  }
 }
'''+s[b:]
a=s.index(' float acc[32]={0};');b=s.index(' if(warp==0) {',a)
s=s[:a]+r'''
 float acc[32]={0};
 #pragma unroll
 for(int n=0;n<8;n++) {
  #pragma unroll
  for(int d=0;d<4;d++) {
   unsigned vr0,vr1;
   unsigned addr=(unsigned)__cvta_generic_to_shared(V+(d*16+lane%16)*64+n*8);
   asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];":"=r"(vr0),"=r"(vr1):"r"(addr));
   asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
    : "+f"(acc[n*4]),"+f"(acc[n*4+1]),"+f"(acc[n*4+2]),"+f"(acc[n*4+3])
    : "r"(pr[d][0]),"r"(pr[d][1]),"r"(pr[d][2]),"r"(pr[d][3]),"r"(vr0),"r"(vr1));
  }
 }
'''+s[b:]
s=s.replace('\n}\nvoid attention','\n } // warp zero arithmetic\n}\nvoid attention')
s=s.replace('<<<grid,128,0,stream>>>','<<<grid,THREADS,0,stream>>>')
for name,threads in [('warp',32),('cooperative',128)]:
 (P/f'{name}.cu').write_text(f'#define THREADS {threads}\n'+s)
 (P/f'{name}.py').write_text((P/'base.py').read_text().replace('.build-base','.build-'+name).replace('codex_sliding_base_rows','codex_sliding_mma_'+name).replace("base.cu",name+'.cu'))
# Hybrid control keeps native FP8 QK WGMMA; only PV and its softmax run in warp zero.
hybrid=base.replace('target=i ^ (((i>>6)&7)<<3)','target=i')
a=hybrid.index(' float acc[32]={0};');b=hybrid.index(' if(warp==0) {',a)
x=s.index(' float acc[32]={0};');y=s.index(' if(warp==0) {',x)
hybrid=hybrid[:a]+s[x:y]+hybrid[b:]
hybrid=hybrid.replace(' float mx[2]',' if(warp==0) {\n float mx[2]')
hybrid=hybrid.replace('\n}\nvoid attention','\n } // warp zero softmax/PV\n}\nvoid attention')
(P/'hybrid.cu').write_text(hybrid)
(P/'hybrid.py').write_text((P/'base.py').read_text().replace('.build-base','.build-hybrid').replace('codex_sliding_base_rows','codex_sliding_mma_hybrid').replace('base.cu','hybrid.cu'))
# Preserve the parent's conflict-avoiding V swizzle for fair matrix controls.
for name in ('warp','cooperative','hybrid'):
 text=(P/f'{name}.cu').read_text()
 text=text.replace('target=i;', 'target=i ^ (((i>>6)&7)<<3);')
 text=text.replace('unsigned addr=(unsigned)__cvta_generic_to_shared(V+(d*16+lane%16)*64+n*8);',
                   'unsigned vi=(d*16+lane%16)*64+n*8;\n   unsigned addr=(unsigned)__cvta_generic_to_shared(V+(vi ^ (((vi>>6)&7)<<3)));')
 assert text!=(P/f'{name}.cu').read_text()
 label=name+'_swizzle'
 (P/f'{label}.cu').write_text(text)
 (P/f'{label}.py').write_text((P/'base.py').read_text().replace('.build-base','.build-'+label).replace('codex_sliding_base_rows','codex_sliding_mma_'+label).replace('base.cu',label+'.cu'))

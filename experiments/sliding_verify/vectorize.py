"""Full L8 tiles have eight times L1's useful query rows; test explicit vector loads."""
from pathlib import Path
R=Path(__file__).resolve().parent
s=(R/'kernel.cu').read_text()
a=s.index('  #pragma unroll\n  for(int i=0;i<16;i++)');b=s.index('  sq+=__shfl',a)
s=s[:a]+'''  #pragma unroll
  for(int i=0;i<4;i++) {
   int col=(i/2)*32+(lane%4)*4+(i%2)*16;
   const __nv_bfloat16* p=qkv+(b*8+token)*width+(h*8+lane/4)*64+col;
   unsigned a,bits;
   asm volatile("ld.global.v2.u32 {%0,%1},[%2];":"=r"(a),"=r"(bits):"l"(p):"memory");
   x[4*i]=__bfloat162float(__ushort_as_bfloat16(a&65535));
   x[4*i+1]=__bfloat162float(__ushort_as_bfloat16(a>>16));
   x[4*i+2]=__bfloat162float(__ushort_as_bfloat16(bits&65535));
   x[4*i+3]=__bfloat162float(__ushort_as_bfloat16(bits>>16));
  }
  #pragma unroll
  for(int i=0;i<16;i++)sq+=x[i]*x[i];
''' +s[b:]
s=s.replace('    #pragma unroll\n    for(int k=0;k<4;k+=2) {', '''    int cscol=d*16+(lane%4)*2+i*8;
    const float* csp=cs+(long)(seq+token)*64+cscol;
    float cc0,cc1,ss0,ss1;
    asm volatile("ld.global.v2.f32 {%0,%1},[%2];":"=f"(cc0),"=f"(cc1):"l"(csp):"memory");
    asm volatile("ld.global.v2.f32 {%0,%1},[%2];":"=f"(ss0),"=f"(ss1):"l"(csp+32):"memory");
    #pragma unroll
    for(int k=0;k<4;k+=2) {''')
s=s.replace('float cc=cs[(long)(seq+token)*64+col/2],ss=cs[(long)(seq+token)*64+col/2+32];','float cc=k==0?cc0:cc1,ss=k==0?ss0:ss1;')
(R/'vector.cu').write_text(s)
s=(R/'native.py').read_text().replace("/'.build'","/'.build-vector'").replace('codex_sliding_verify','codex_sliding_verify_vector').replace("ROOT/'kernel.cu'","ROOT/'vector.cu'")
(R/'vector.py').write_text(s)

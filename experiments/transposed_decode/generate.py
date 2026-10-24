from pathlib import Path

PACKED = '''   for(int j=0;j<4;j++){
    unsigned i=8*tid+j*1024,target=sw128(i);
    unsigned long long bytes=*(unsigned long long*)(VR[slot][h]+i);unsigned cv[4];
    #pragma unroll
    for(int z=0;z<4;z++) {__half2_raw v=__nv_cvt_fp8x2_to_halfraw2((unsigned short)(bytes>>(16*z)),__NV_E4M3);cv[z]=*(unsigned*)&v;}
    asm volatile("st.shared.v4.u32 [%4],{%0,%1,%2,%3};"::"r"(cv[0]),"r"(cv[1]),"r"(cv[2]),"r"(cv[3]),"r"((unsigned)__cvta_generic_to_shared(VT[h]+target)):"memory");
   }
'''

def derive(s):
    s=s.replace(' %4,%5,1,1,1,0,0;',' %4,%5,1,1,1,1,0;')
    a=s.index('   for(int j=0;j<32;j++){')
    b=s.index('   #pragma unroll\n   for(int c=0;c<2;c++){',a)
    s=s[:a]+PACKED+s[b:]
    s=s.replace('desc128(VT[h]+sw128(d*16))','((((unsigned long)__cvta_generic_to_shared(VT[h]+d*1024)&0x3ffff)>>4)|(64ull<<16)|(64ull<<32)|(1ull<<62))')
    return s

if __name__=='__main__':
    root=Path(__file__).resolve().parent
    (root/'natural_v.cu').write_text(derive((root/'kernel.cu').read_text()))

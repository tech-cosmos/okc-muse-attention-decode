"""Generate isolated instruction changes from the session-3 native source."""
from pathlib import Path
root = Path(__file__).resolve().parent
base = (root/'baseline.py').read_text()

def x4(s):
    old = '''       #pragma unroll
       for(int j=0;j<2;j++) {
         unsigned addr=__cvta_generic_to_shared(V[wg]+(d*16+lane%16)*(D+8)+n*16+j*8);
         unsigned v0,v1;
         asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];":"=r"(v0),"=r"(v1):"r"(addr));'''
    new = '''       unsigned addr=__cvta_generic_to_shared(V[wg]+(d*16+lane%16)*(D+8)+n*16+(lane/16)*8);
       unsigned vv[4];
       asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];":"=r"(vv[0]),"=r"(vv[1]),"=r"(vv[2]),"=r"(vv[3]):"r"(addr));
       #pragma unroll
       for(int j=0;j<2;j++) {
         unsigned v0=vv[j*2],v1=vv[j*2+1];'''
    assert old in s
    return s.replace(old,new)

def firstwrite(s):
    start=s.index('   wmma::fill_fragment(score[0]')
    end=s.index('   asm volatile("wgmma.fence',start)
    s=s[:start]+s[end:]
    start=s.index('     asm volatile("wgmma.mma_async')
    end=s.index('\n   }\n   asm volatile("wgmma.commit',start)
    asm=s[start:end]
    first=asm.replace(' : "+f"',' : "=f"').replace(', "+f"',', "=f"').replace('%20, 1, 1, 1;', '%20, 0, 1, 1;')
    assert '"+f"' not in first
    return s[:start]+'     if(d==0) {\n'+first+'\n     } else {\n'+asm+'\n     }'+s[end:]

def vector(s):
    start=s.index('   #pragma unroll\n   for(unsigned j=0;j<N*D/256;j++)')
    end=s.index('   consumer_barrier(wg);',start)
    replacement='''   #pragma unroll
   for(unsigned j=0;j<N*D/1024;j++) {
     unsigned i=8*tid+j*1024;
     unsigned target=(i/D)*(D+8)+i%D;
     unsigned long long bytes=*(const unsigned long long*)(raw[stage][wg][1]+(i/D)*(D+16)+i%D);
     unsigned converted[4];
     #pragma unroll
     for(int k=0;k<4;k++) {
       __half2_raw vv=__nv_cvt_fp8x2_to_halfraw2((unsigned short)(bytes>>(16*k)),__NV_E4M3);
       converted[k]=*(unsigned*)&vv;
     }
     *(uint4*)(V[wg]+target)=make_uint4(converted[0],converted[1],converted[2],converted[3]);
   }
'''
    return s[:start]+replacement+s[end:]

def forced(s):
    old='     *(uint4*)(V[wg]+target)=make_uint4(converted[0],converted[1],converted[2],converted[3]);'
    new='''     asm volatile("st.shared.v4.u32 [%4], {%0,%1,%2,%3};"::"r"(converted[0]),"r"(converted[1]),"r"(converted[2]),"r"(converted[3]),"r"((unsigned)__cvta_generic_to_shared(V[wg]+target)):"memory");'''
    assert old in s
    return s.replace(old,new)

for name,source in [('x4',x4(base)),('firstwrite',firstwrite(base)),('combined',firstwrite(x4(base))),
                    ('vector',vector(base)),('all',vector(firstwrite(x4(base)))),
                    ('forced',forced(vector(x4(base))))]:
    source=source.replace("'.build-pair-mbar'",repr('.build-'+name)).replace("name='codex_verify_pair3mbar'",f"name='codex_matrix_{name}'")
    (root/(name+'.py')).write_text(source)

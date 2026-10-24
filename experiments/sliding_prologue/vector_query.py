from pathlib import Path
ROOT=Path(__file__).resolve().parent
s=(ROOT/'direct.cu').read_text()
a=s.index('  #pragma unroll\n  for(int i=0;i<16;i++)');z=s.index('  sq+=__shfl',a)
s=s[:a]+r'''
  #pragma unroll
  for(int j=0;j<4;j++) {
   int col=(j/2)*32+(lane%4)*4+(j%2)*16;
   uint2 bits=*(const uint2*)(qkv+b*width+(h*8+lane/4)*64+col);
   #pragma unroll
   for(int k=0;k<4;k++) {
    unsigned short v=(k<2?bits.x:bits.y)>>(16*(k%2));
    __nv_bfloat16_raw raw;raw.x=v;
    x[4*j+k]=__bfloat162float(raw);sq+=x[4*j+k]*x[4*j+k];
   }
  }
''' + s[z:]
(ROOT/'vector.cu').write_text(s)
p=(ROOT/'direct.py').read_text().replace("ROOT/'.build-direct'","ROOT/'.build-vector'").replace("codex_sliding_direct'","codex_sliding_vector'").replace("ROOT/'direct.cu'","ROOT/'vector.cu'")
(ROOT/'vector.py').write_text(p)
p=(ROOT/'compare.py').read_text().replace('import native,direct','import native,direct,vector')
p=p.replace("names=['production','late','early','direct_late','direct_early']","names=['production','direct_early','vector_early','vector_late','direct_late']")
p=p.replace("m=direct if variant.startswith('direct') else native","m=vector if variant.startswith('vector') else direct")
p=p.replace('results/comparison-direct.json','results/comparison-vector.json')
(ROOT/'compare_vector.py').write_text(p)

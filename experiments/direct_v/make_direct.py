from pathlib import Path
p=Path(__file__).parent
s=(p/'baseline.py').read_text()
s=s.replace('Slower than submission.py; never dispatched or submitted. See README.txt.', 'Session-6 direct FP8-register V operand experiment; not production.')
s=s.replace(' __shared__ __align__(32) half V[2][N*(D+8)];','')
a=s.index('   #pragma unroll\n   for(unsigned j=0;j<N*D/256;j++)')
b=s.index('   wmma::fragment<wmma::accumulator,16,16,16,float> score[2];',a)
s=s[:a]+s[b:]
a=s.index('   #pragma unroll\n   for(int n=0;n<4;n++) {\n     #pragma unroll\n     for(int d=0;d<2;d++) {')
b=s.index('   consumer_barrier(wg);',a)
s=s[:a]+r'''
   #pragma unroll
   for(int n=0;n<4;n++) {
     #pragma unroll
     for(int d=0;d<2;d++) {
       // Each b16 matrix element is a pair of adjacent FP8 columns.
       // ldmatrix transposes these pairs; shuffles select the source column
       // pair and byte permutes select one FP8 column, across two K rows.
       unsigned addr=__cvta_generic_to_shared(raw[stage][wg][1]+(d*16+lane%16)*(D+16)+n*16);
       unsigned bytes0,bytes1;
       asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
          :"=r"(bytes0),"=r"(bytes1):"r"(addr));
       #pragma unroll
       for(int j=0;j<2;j++) {
         unsigned col=lane/4,src=((col/2)+j*4)*4+lane%4;
         unsigned a=__shfl_sync(0xffffffff,bytes0,src);
         unsigned b=__shfl_sync(0xffffffff,bytes1,src);
         unsigned select=(col&1)?0x7731:0x7720;
         unsigned short ab0=__byte_perm(a,0,select),ab1=__byte_perm(b,0,select);
         __half2_raw x0=__nv_cvt_fp8x2_to_halfraw2(ab0,__NV_E4M3);
         __half2_raw x1=__nv_cvt_fp8x2_to_halfraw2(ab1,__NV_E4M3);
         unsigned v0=*(unsigned*)&x0,v1=*(unsigned*)&x1;
         float* z=acc[n].x+j*4;
         asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
           : "+f"(z[0]),"+f"(z[1]),"+f"(z[2]),"+f"(z[3])
           : "r"(pr[d][0]),"r"(pr[d][1]),"r"(pr[d][2]),"r"(pr[d][3]),"r"(v0),"r"(v1));
       }
     }
   }
''' + s[b:]
s=s.replace("'.build-pair-mbar'","'.build-direct'").replace("name='codex_verify_pair3mbar'","name='codex_verify_direct_v'")
(p/'direct.py').write_text(s)

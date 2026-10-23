from pathlib import Path
p=Path(__file__).parent
s=(p/'baseline.py').read_text()
s=s.replace('template<int STAGES, int PAGE>', 'template<int STAGES, int PAGE, int PROD>')
s=s.replace('__launch_bounds__(320,2)', '__launch_bounds__(256+PROD,2)')
s=s.replace('unsigned char raw[STAGES][2][2][N*(D+16)]', 'unsigned char raw[STAGES][2][N*(D+16)]')
s=s.replace('half V[2][N*(D+8)]', 'half V[STAGES][2][N*(D+8)]')
s=s.replace('ready[STAGES],empty[STAGES]', 'ready[STAGES],empty[STAGES],values_ready[STAGES]')
s=s.replace('asm volatile("mbarrier.init.shared.b64 [%0],64;"::"r"((unsigned)__cvta_generic_to_shared(ready+global_tid)):"memory");', '''asm volatile("mbarrier.init.shared.b64 [%0],%1;"::"r"((unsigned)__cvta_generic_to_shared(ready+global_tid)),"r"(PROD):"memory");
  asm volatile("mbarrier.init.shared.b64 [%0],%1;"::"r"((unsigned)__cvta_generic_to_shared(values_ready+global_tid)),"r"(PROD):"memory");''')
s=s.replace('i+=64)', 'i+=PROD)')
s=s.replace('raw[stage][hh][0]', 'raw[stage][hh]')
s=s.replace('raw[stage][wg][0]', 'raw[stage][wg]')
s=s.replace('copy16(raw[stage][hh][1]+row_in_tile*(D+16)+piece*16,vc+row+(i%8)*16);', '''// Convert an entire aligned 16-byte cache piece in its producer.
           uint4 bits=*(const uint4*)(vc+row+(i%8)*16);
           unsigned words[4]={bits.x,bits.y,bits.z,bits.w};
           #pragma unroll
           for(int wi=0;wi<4;wi++) {
             __half2_raw a=__nv_cvt_fp8x2_to_halfraw2((unsigned short)words[wi],__NV_E4M3);
             __half2_raw z=__nv_cvt_fp8x2_to_halfraw2((unsigned short)(words[wi]>>16),__NV_E4M3);
             *(half2*)(V[stage][hh]+row_in_tile*(D+8)+piece*16+wi*4)=a;
             *(half2*)(V[stage][hh]+row_in_tile*(D+8)+piece*16+wi*4+2)=z;
           }''')
s=s.replace('raw[stage][hh][1][(ii/D)*(D+16)+ii%D]=ii<8*D?own[hh][1][ii]:0;', 'V[stage][hh][(ii/D)*(D+8)+ii%D]=ii<8*D?from8(own[hh][1][ii]):__float2half(0.f);')
s=s.replace('       if(isnew) {\n         asm volatile("bar.sync 3,64;":::"memory");', '       mbarrive(values_ready+stage);\n       if(isnew) {\n         asm volatile("bar.sync 3,%0;"::"r"(PROD):"memory");')
a=s.index('   #pragma unroll\n   for(unsigned j=0;j<N*D/256;j++)')
b=s.index('   wmma::fragment<wmma::accumulator,16,16,16,float> score[2];',a)
s=s[:a]+'   mbwait(values_ready+stage,(tile/STAGES)&1);\n'+s[b:]
s=s.replace('V[wg]+', 'V[stage][wg]+')
s=s.replace('double scale, int stages)', 'double scale, int stages, int producers)')
s=s.replace('double scale, int stages);', 'double scale, int stages, int producers);')
s=s.replace('#define LAUNCH(S,P) verify<S,P><<<grid,320,0,stream>>>', '#define LAUNCH(S,P,R) verify<S,P,R><<<grid,256+R,0,stream>>>')
a=s.index(' if(page==1) {',s.index('void attention(at::Tensor'))
b=s.index(' C10_CUDA_KERNEL_LAUNCH_CHECK();',a)
s=s[:a]+''' #define BY_STAGE(P,R) if(stages==2){LAUNCH(2,P,R);}else if(stages==3){LAUNCH(3,P,R);}else if(stages==4){LAUNCH(4,P,R);}else{LAUNCH(5,P,R);}
 if(page==1) {
  if(producers==2){BY_STAGE(1,64);}else{BY_STAGE(1,128);}
 }else{
  if(producers==2){BY_STAGE(64,64);}else{BY_STAGE(64,128);}
 }
'''+s[b:]
s=s.replace("'.build-pair-mbar'", "'.build-producer-convert'")
s=s.replace("name='codex_verify_pair3mbar'", "name='codex_verify_producer_convert'")
s=s.replace('def run(data, splits=None, stages=3):', 'def run(data, splits=None, stages=3, producers=2):')
s=s.replace('    assert stages in (2,3,4,5)', '    assert stages in (2,3,4,5) and producers in (2,4)')
s=s.replace('submission.LOG2E,stages)', 'submission.LOG2E,stages,producers)')
s=s.replace('Two producer warps, two four-warp attention consumer groups, full-row FP8 gathers.', 'Producer warps gather K asynchronously and convert aligned V vectors directly to staged FP16.\nTwo completion barriers order K copies and generic V stores independently.')
(p/'producer.py').write_text(s)

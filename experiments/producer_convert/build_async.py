from pathlib import Path
p=Path(__file__).parent
s=(p/'baseline.py').read_text()
s=s.replace('template<int STAGES, int PAGE>', 'template<int STAGES, int PAGE, int CONV>')
s=s.replace('__launch_bounds__(320,2)', '__launch_bounds__(320+CONV,2)')
s=s.replace('half V[2][N*(D+8)]', 'half V[STAGES][2][N*(D+8)]')
s=s.replace('ready[STAGES],empty[STAGES]', 'ready[STAGES],empty[STAGES],values_ready[STAGES]')
old='asm volatile("mbarrier.init.shared.b64 [%0],64;"::"r"((unsigned)__cvta_generic_to_shared(ready+global_tid)):"memory");'
s=s.replace(old,old+'\n  asm volatile("mbarrier.init.shared.b64 [%0],%1;"::"r"((unsigned)__cvta_generic_to_shared(values_ready+global_tid)),"r"(CONV):"memory");')
s=s.replace(' if(wg>=2) {',' if(global_tid>=256 && global_tid<320) {')
a=s.index(' wmma::fragment<wmma::accumulator,16,16,16,float> acc[4];')
s=s[:a]+''' // Two dedicated conversion warps allow the asynchronous gather producer to run ahead.
 if(global_tid>=320) {
  unsigned converter=global_tid-320;
  for(int tile=0;tile<tiles;tile++) {
   int stage=tile%STAGES;
   mbwait(ready+stage,(tile/STAGES)&1);
   #pragma unroll
   for(unsigned j=0;j<(2*N*D)/(2*CONV);j++) {
    unsigned i=2*converter+j*(2*CONV),hh=i/(N*D),ii=i%(N*D);
    __half2_raw vv=__nv_cvt_fp8x2_to_halfraw2(*(unsigned short*)(raw[stage][hh][1]+(ii/D)*(D+16)+ii%D),__NV_E4M3);
    *(half2*)(V[stage][hh]+(ii/D)*(D+8)+ii%D)=vv;
   }
   mbarrive(values_ready+stage);
  }
  return;
 }
'''+s[a:]
a=s.index('   #pragma unroll\n   for(unsigned j=0;j<N*D/256;j++)')
b=s.index('   wmma::fragment<wmma::accumulator,16,16,16,float> score[2];',a)
s=s[:a]+'   mbwait(values_ready+stage,(tile/STAGES)&1);\n'+s[b:]
s=s.replace('V[wg]+', 'V[stage][wg]+')
s=s.replace('double scale, int stages)', 'double scale, int stages, int converters)')
s=s.replace('#define LAUNCH(S,P) verify<S,P><<<grid,320,0,stream>>>', '#define LAUNCH(S,P,C) verify<S,P,C><<<grid,320+C,0,stream>>>')
a=s.index(' if(page==1) {',s.index('void attention(at::Tensor'))
b=s.index(' C10_CUDA_KERNEL_LAUNCH_CHECK();',a)
s=s[:a]+''' #define BY_STAGE(P,C) if(stages==2){LAUNCH(2,P,C);}else if(stages==3){LAUNCH(3,P,C);}else{LAUNCH(4,P,C);}
 if(page==1) {
  if(converters==2){BY_STAGE(1,64);}else{BY_STAGE(1,128);}
 }else{
  if(converters==2){BY_STAGE(64,64);}else{BY_STAGE(64,128);}
 }
'''+s[b:]
s=s.replace('def run(data, splits=None, stages=3):', 'def run(data, splits=None, stages=3, converters=2):')
s=s.replace('submission.LOG2E,stages)', 'submission.LOG2E,stages,converters)')
# Five stages exceed static shared-memory limit; only 2,3,4 compile.
s=s.replace('else if(stages==5) { LAUNCH(5,1); }','')
s=s.replace('else if(stages==5) { LAUNCH(5,64); }','')
s=s.replace('assert stages in (2,3,4,5)', 'assert stages in (2,3,4) and converters in (2,4)')
s=s.replace("'.build-pair-mbar'", "'.build-async-convert-v2'")
s=s.replace("name='codex_verify_pair3mbar'", "name='codex_verify_async_convert_v2'")
s=s.replace('Two producer warps, two four-warp attention consumer groups, full-row FP8 gathers.', 'Two gather producer warps, two conversion warps, and eight attention consumer warps.\nConversion runs ahead in the shared-memory ring and publishes its own completion barrier.')
(p/'async_convert.py').write_text(s)

from pathlib import Path
s=Path('experiments/gather/pair_p2.py').read_text()
s=s.replace('template<int STAGES, int PAGE>','''__device__ __forceinline__ void mbwait(unsigned long long* p,unsigned phase) {
 unsigned done;
 do {asm volatile("{.reg .pred p; mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p,[%1],%2; selp.b32 %0,1,0,p;}" : "=r"(done) : "r"((unsigned)__cvta_generic_to_shared(p)),"r"(phase):"memory");}while(!done);
}
__device__ __forceinline__ void mbarrive(unsigned long long* p) {
 asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _,[%0];"::"r"((unsigned)__cvta_generic_to_shared(p)):"memory");
}
template<int STAGES, int PAGE>''')
s=s.replace('__shared__ unsigned ready[STAGES][2];','__shared__ unsigned long long ready[STAGES],empty[STAGES];')
s=s.replace('for(int i=global_tid;i<STAGES*2;i+=blockDim.x) ((unsigned*)ready)[i]=0;','''if(global_tid<STAGES) {
  asm volatile("mbarrier.init.shared.b64 [%0],64;"::"r"((unsigned)__cvta_generic_to_shared(ready+global_tid)):"memory");
  asm volatile("mbarrier.init.shared.b64 [%0],2;"::"r"((unsigned)__cvta_generic_to_shared(empty+global_tid)):"memory");
 }''')
a=s.index('   for(int step=0;step<tiles+STAGES-1;step++) {')
b=s.index('       bool isnew=',a)
s=s[:a]+'''   for(int step=0;step<tiles;step++) {
       int stage=step%STAGES;
       if(step>=STAGES)mbwait(empty+stage,((step/STAGES)-1)&1);
'''+s[b:]
s=s.replace('''       asm volatile("cp.async.commit_group;":::"memory");
     }
   }
   return;''','''       if(isnew) {
         asm volatile("bar.sync 3,64;":::"memory");
         asm volatile("fence.proxy.async.shared::cta;":::"memory");
         mbarrive(ready+stage);
       }else{
         asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];"::"r"((unsigned)__cvta_generic_to_shared(ready+stage)):"memory");
       }
   }
   return;''')
s=s.replace('while(acquire(&ready[stage][wg])!=1) {}\n   asm volatile("membar.cta;":::"memory");','mbwait(ready+stage,(tile/STAGES)&1);')
s=s.replace('if(tid==0) release(&ready[stage][wg],0);','if(tid==0)mbarrive(empty+stage);')
s=s.replace("'.build-pair-p2'","'.build-pair-mbar'").replace('codex_verify_pair3p2','codex_verify_pair3mbar')
s=s.replace('else if(stages==5) { LAUNCH(3,','else if(stages==5) { LAUNCH(5,')
Path('experiments/gather/pair_mbar.py').write_text(s)
s=Path('experiments/gather/bench_pair_p2.py').read_text().replace('pair_p2','pair_mbar').replace('pair-p2-','pair-mbar-').replace('(3,4),(132,264,528,1056)','(3,4,5),(132,264,528)')
Path('experiments/gather/bench_pair_mbar.py').write_text(s)

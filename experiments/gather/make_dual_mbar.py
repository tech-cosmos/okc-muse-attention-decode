from pathlib import Path
s=Path('experiments/gather/dual_page.py').read_text()
r=Path('experiments/gather/pair_mbar.py').read_text()
a=r.index('__device__ __forceinline__ void mbwait');b=r.index('template<int STAGES',a)
s=s.replace('template<int STAGES, int PAGE>',r[a:b]+'template<int STAGES, int PAGE>')
s=s.replace('__shared__ unsigned ready[STAGES];','__shared__ unsigned long long ready[STAGES],empty[STAGES];')
s=s.replace('if(tid<STAGES)ready[tid]=0;','''if(tid<STAGES) {
 asm volatile("mbarrier.init.shared.b64 [%0],32;"::"r"((unsigned)__cvta_generic_to_shared(ready+tid)):"memory");
 asm volatile("mbarrier.init.shared.b64 [%0],1;"::"r"((unsigned)__cvta_generic_to_shared(empty+tid)):"memory");
 }''')
a=s.index('   for(int step=0;step<tiles+STAGES-1;step++) {');b=s.index('       bool isnew=',a)
s=s[:a]+'''   for(int step=0;step<tiles;step++) {
       int stage=step%STAGES;
       if(step>=STAGES)mbwait(empty+stage,((step/STAGES)-1)&1);
'''+s[b:]
s=s.replace('''       asm volatile("cp.async.commit_group;":::"memory");
     }
   }
   return;''','''       if(isnew) {
         __syncwarp();
         asm volatile("fence.proxy.async.shared::cta;":::"memory");
         mbarrive(ready+stage);
       }else{
         asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];"::"r"((unsigned)__cvta_generic_to_shared(ready+stage)):"memory");
       }
   }
   return;''')
s=s.replace('while(acquire(ready+stage)!=1) {}\n   asm volatile("membar.cta;":::"memory");','mbwait(ready+stage,(tile/STAGES)&1);')
s=s.replace('if(tid==0) release(ready+stage,0);','if(tid==0)mbarrive(empty+stage);')
s=s.replace("'.build-dual-page'","'.build-dual-mbar'").replace('codex_verify_dual3page','codex_verify_dual3mbar')
Path('experiments/gather/dual_mbar.py').write_text(s)
s=Path('experiments/gather/bench_dual.py').read_text().replace('import dual as','import dual_mbar as').replace('results/dual-','results/dual-mbar-')
Path('experiments/gather/bench_dual_mbar.py').write_text(s)

from pathlib import Path
s=Path('experiments/gather/gather.py').read_text()
r=Path('experiments/gather/pair_mbar.py').read_text();a=r.index('__device__ __forceinline__ void mbwait');b=r.index('template<int STAGES',a)
s=s.replace('template<int H,int ST,int PW>',r[a:b]+'''__device__ void bulk(void* d,const void* s,unsigned bytes,unsigned long long* bar){
 asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1],%2,[%3];"::"r"((unsigned)__cvta_generic_to_shared(d)),"l"(s),"r"(bytes),"r"((unsigned)__cvta_generic_to_shared(bar)):"memory");
}
template<int H,int ST,int PW>''')
s=s.replace('__shared__ unsigned ready[ST];','__shared__ unsigned long long ready[ST],empty[ST];')
s=s.replace('if(tid<ST)ready[tid]=0;', '''if(tid<ST){
 asm volatile("mbarrier.init.shared.b64 [%0],1;"::"r"((unsigned)__cvta_generic_to_shared(ready+tid)):"memory");
 asm volatile("mbarrier.init.shared.b64 [%0],1;"::"r"((unsigned)__cvta_generic_to_shared(empty+tid)):"memory");
 }''')
a=s.index(' if(PROD && tid>=THREADS){');b=s.index(' out[',a)
s=s[:a]+''' if(tid>=THREADS){
  for(int step=0;step<tiles;step++) {
   int stage=step%ST;
   if(step>=ST)mbwait(empty+stage,((step/ST)-1)&1);
   if(pload==0)asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _,[%0],%1;"::"r"((unsigned)__cvta_generic_to_shared(ready+stage)),"r"(2*N*D):"memory");
   if(PW==1)__syncwarp();else asm volatile("bar.sync 2,%0;"::"r"(PW*32):"memory");
   for(int i=pload;i<N;i+=PW*32){
    int t=min(lo+step*N+i,hi-1);
    long row=((long)pt[b*MP+t/PAGE]*PAGE+t%PAGE)*HKV*64+h*64;
    bulk(raw[stage][0]+i*D,k+row,D,ready+stage);
    bulk(raw[stage][1]+i*D,v+row,D,ready+stage);
   }
  }
  return;
 }
 for(int tile=0;tile<tiles;tile++){
  int st=tile%ST;mbwait(ready+st,(tile/ST)&1);
  for(int i=tid;i<N*D/4;i+=THREADS){
   if(lo+tile*N+i/(D/4)<hi)sum+=((unsigned*)raw[st][0])[i]+((unsigned*)raw[st][1])[i];
  }
  asm volatile("bar.sync 1,128;":::"memory");if(tid==0)mbarrive(empty+st);
 }
'''+s[b:]
s=s.replace('RUN(H,ST,0)','RUN(H,ST,1)').replace("'.build-gather'","'.build-gather-bulk'").replace('codex_gather3b','codex_gather3bulk')
Path('experiments/gather/gather_bulk.py').write_text(s)
s=Path('experiments/gather/bench_gather.py').read_text().replace('import gather','import gather_bulk as gather').replace('(2,),(2,4),(2,3,4),(528,1056,1584)','(2,),(1,),(2,3,4,6),(528,1056,1584)').replace('gather-multi-','gather-bulk-')
Path('experiments/gather/bench_gather_bulk.py').write_text(s)

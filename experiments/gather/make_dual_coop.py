from pathlib import Path
s=Path('experiments/gather/dual.py').read_text()
s=s.replace('__launch_bounds__(160,3)','__launch_bounds__(128,4)').replace('<<<grid,160,0,stream>>>','<<<grid,128,0,stream>>>').replace('row+=5','row+=4')
a=s.index(' if(warp==4) {');b=s.index(' wmma::fragment<wmma::accumulator',a)
r=Path('experiments/gather/pair_coop.py').read_text()
x=r.index(' auto enqueue');y=r.index(' wmma::fragment<wmma::accumulator',x)
block=r[x:y].replace('global_tid','tid').replace('i+=256','i+=128')
s=s[:a]+block+s[b:]
s=s.replace('   while(acquire(ready+stage)!=1) {}\n   asm volatile("membar.cta;":::"memory");','''   if(tile+STAGES-1<tiles) {asm volatile("cp.async.wait_group %0;"::"n"(STAGES-2):"memory");}
   else {asm volatile("cp.async.wait_group 0;":::"memory");}
   asm volatile("fence.proxy.async.shared::cta;":::"memory");
   __syncthreads();''')
s=s.replace('   if(tid==0) release(ready+stage,0);','   __syncthreads();\n   if(tile+STAGES-1<tiles)enqueue(tile+STAGES-1);')
s=s.replace("'.build-dual'","'.build-dual-coop'").replace('codex_verify_dual3','codex_verify_dual3coop')
Path('experiments/gather/dual_coop.py').write_text(s)
s=Path('experiments/gather/bench_dual.py').read_text().replace('import dual as','import dual_coop as').replace('results/dual-','results/dual-coop-')
Path('experiments/gather/bench_dual_coop.py').write_text(s)
# Pointer-prefetch producer variant, one page-table load per row, broadcast via shuffles.
s=Path('experiments/gather/dual.py').read_text()
s=s.replace('''         for(int i=lane;i<N*8;i+=32) {
           unsigned row_in_tile=i/8, hh=(i%8)/4, piece=i%4;
           int t=min(lo+step*N+row_in_tile,hi-1);
           long row=((long)pt[b*MP+t/PAGE]*PAGE+t%PAGE)*HKV*D+blockIdx.z*2*D;''','''         int myt=min(lo+step*N+lane,hi-1);
         unsigned myrow=pt[b*MP+myt/PAGE]*PAGE+myt%PAGE;
         #pragma unroll
         for(int i=lane;i<N*8;i+=32) {
           unsigned row_in_tile=i/8, hh=(i%8)/4, piece=i%4;
           unsigned rr=__shfl_sync(0xffffffff,myrow,row_in_tile);
           long row=(long)rr*HKV*D+blockIdx.z*2*D;''')
s=s.replace("'.build-dual'","'.build-dual-page'").replace('codex_verify_dual3','codex_verify_dual3page')
Path('experiments/gather/dual_page.py').write_text(s)
s=Path('experiments/gather/bench_dual.py').read_text().replace('import dual as','import dual_page as').replace('results/dual-','results/dual-page-')
Path('experiments/gather/bench_dual_page.py').write_text(s)

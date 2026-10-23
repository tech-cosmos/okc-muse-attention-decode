from pathlib import Path
s=Path('experiments/gather/pair_v1.py').read_text()
s=s.replace('__launch_bounds__(288,2)','__launch_bounds__(256,2)').replace('<<<grid,288,0,stream>>>','<<<grid,256,0,stream>>>')
s=s.replace('asm volatile("bar.sync %0, 128;"::"r"(wg+1):"memory");', 'if(wg==0)asm volatile("bar.sync 1,128;":::"memory"); else asm volatile("bar.sync 2,128;":::"memory");')
start=s.index(' if(wg==2) {')
end=s.index(' wmma::fragment<wmma::accumulator',start)
s=s[:start]+''' auto enqueue = [&](int step) {
    int stage=step%STAGES;
    bool isnew=split==0 && step==tiles-1;
    if(!isnew) {
      for(int i=global_tid;i<N*8;i+=256) {
        unsigned row_in_tile=i/8, hh=(i%8)/4, piece=i%4;
        int t=min(lo+step*N+row_in_tile,hi-1);
        long row=((long)pt[b*MP+t/PAGE]*PAGE+t%PAGE)*HKV*D+blockIdx.z*2*D;
        copy16(raw[stage][hh][0]+(row_in_tile/8)*512+piece*128+(row_in_tile%8)*16,kc+row+(i%8)*16);
        copy16(raw[stage][hh][1]+row_in_tile*(D+16)+piece*16,vc+row+(i%8)*16);
      }
    } else {
      for(int i=global_tid;i<2*N*D;i+=256) {
        unsigned hh=i/(N*D), ii=i%(N*D);
        raw[stage][hh][0][(ii/(D*8))*512+((ii%D)/16)*128+((ii/D)%8)*16+ii%16]=ii<8*D?own[hh][0][ii]:0;
        raw[stage][hh][1][(ii/D)*(D+16)+ii%D]=ii<8*D?own[hh][1][ii]:0;
      }
    }
    asm volatile("cp.async.commit_group;":::"memory");
 };
 for(int step=0;step<min(tiles,STAGES-1);step++) enqueue(step);
'''+s[end:]
s=s.replace('   while(acquire(&ready[stage][wg])!=1) {}\n   asm volatile("membar.cta;":::"memory");','''   if(tile+STAGES-1<tiles) {asm volatile("cp.async.wait_group %0;"::"n"(STAGES-2):"memory");}
   else {asm volatile("cp.async.wait_group 0;":::"memory");}
   asm volatile("fence.proxy.async.shared::cta;":::"memory");
   __syncthreads();''')
s=s.replace('   if(tid==0) release(&ready[stage][wg],0);','   __syncthreads();\n   if(tile+STAGES-1<tiles)enqueue(tile+STAGES-1);')
s=s.replace("'.build-pair'","'.build-pair-coop'").replace("name='codex_verify_pair3'","name='codex_verify_pair3coop'")
s=s.replace('else if(stages==5) { LAUNCH(3,','else if(stages==5) { LAUNCH(5,').replace('else if(stages==6) { LAUNCH(3,','else if(stages==6) { LAUNCH(6,').replace('else if(stages==7) { LAUNCH(3,','else if(stages==7) { LAUNCH(7,')
Path('experiments/gather/pair_coop.py').write_text(s)
s=Path('experiments/gather/bench_pair.py').read_text().replace('import pair','import pair_coop as pair').replace('results/pair-{batch}','results/pair-coop-{batch}').replace('(1,16,32)','(16,32)').replace('(2,3),(132,264,396,528,792)','(3,4,6),(132,264,528,1056)')
Path('experiments/gather/bench_pair_coop.py').write_text(s)
s=Path('experiments/gather/pair_p4.py').read_text().replace('__launch_bounds__(384,2)','__launch_bounds__(320,2)').replace('<<<grid,384,0,stream>>>','<<<grid,320,0,stream>>>').replace('bar.sync 3, 128','bar.sync 3, 64').replace('i+=128)','i+=64)').replace("'.build-pair-p4'","'.build-pair-p2'").replace('codex_verify_pair3p4','codex_verify_pair3p2')
Path('experiments/gather/pair_p2.py').write_text(s)
s=Path('experiments/gather/bench_pair_p4.py').read_text().replace('pair_p4','pair_p2').replace('pair-p4-','pair-p2-')
Path('experiments/gather/bench_pair_p2.py').write_text(s)

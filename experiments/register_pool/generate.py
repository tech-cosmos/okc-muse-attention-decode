from pathlib import Path
p=Path(__file__).resolve().parent
src=(p/'baseline.py').read_text()
# Use a complete producer warpgroup for legal warpgroup register repartition.
src=src.replace('__launch_bounds__(320,2)','__launch_bounds__(384,3)')
src=src.replace('<<<grid,320,','<<<grid,384,')
src=src.replace('mbarrier.init.shared.b64 [%0],64;', 'mbarrier.init.shared.b64 [%0],128;')
src=src.replace('i+=64)', 'i+=128)')
src=src.replace('bar.sync 3,64;', 'bar.sync 3,128;')
src=src.replace('const int b=blockIdx.y,', '''if(wg>=2) {
   asm volatile("setmaxnreg.dec.sync.aligned.u32 24;":::"memory");
 }else{
   asm volatile("setmaxnreg.inc.sync.aligned.u32 72;":::"memory");
 }
 const int b=blockIdx.y,''')
src=src.replace("'.build-pair-mbar'", "'.build-dynamic'").replace("name='codex_verify_pair3mbar'", "name='codex_verify_regpool'")
src=src.replace('s=splits or max(1,264//', 's=splits or max(1,396//')
(p/'dynamic.py').write_text(src)
# Same four-producer warpgroup and occupancy target, without register redistribution.
control=src.replace('''if(wg>=2) {
   asm volatile("setmaxnreg.dec.sync.aligned.u32 24;":::"memory");
 }else{
   asm volatile("setmaxnreg.inc.sync.aligned.u32 72;":::"memory");
 }
 ''','').replace("'.build-dynamic'", "'.build-static'").replace("name='codex_verify_regpool'", "name='codex_verify_staticregs'")
(p/'static.py').write_text(control)
# Initial 56 registers: 2 consumer groups at 64, producer at 40.
valid=src.replace('dec.sync.aligned.u32 24','dec.sync.aligned.u32 40').replace('inc.sync.aligned.u32 72','inc.sync.aligned.u32 64').replace("'.build-dynamic'", "'.build-pool3'").replace("name='codex_verify_regpool'", "name='codex_verify_pool3'")
(p/'pool3.py').write_text(valid)
# Initial 80 registers: 2 consumer groups at 96, producer at 48.
valid2=src.replace('__launch_bounds__(384,3)','__launch_bounds__(384,2)').replace('dec.sync.aligned.u32 24','dec.sync.aligned.u32 48').replace('inc.sync.aligned.u32 72','inc.sync.aligned.u32 96').replace("'.build-dynamic'", "'.build-pool2'").replace("name='codex_verify_regpool'", "name='codex_verify_pool2'").replace('396//','264//')
(p/'pool2.py').write_text(valid2)
control2=control.replace('__launch_bounds__(384,3)','__launch_bounds__(384,2)').replace("'.build-static'", "'.build-static2'").replace("name='codex_verify_staticregs'", "name='codex_verify_static2'").replace('396//','264//')
(p/'static2.py').write_text(control2)
# Remove the eight query registers that would otherwise live across the PV loop.
# Volatile shared loads deliberately rematerialize each 32-wide Q fragment.
qstart=valid.index(' unsigned q[2][4];')
qend=valid.index(' for(int tile=0;',qstart)
reload=valid[:qstart]+valid[qend:]
qload='''     unsigned q[4];
     #pragma unroll
     for(int i=0;i<4;i++) {
       unsigned row=warp*16+lane/4+(i%2)*8, col=d*32+(lane%4)*4+(i/2)*16;
       q[i]=*(volatile unsigned*)(Q[wg]+row*(D+16)+col);
     }
'''
reload=reload.replace('     unsigned long desc=',qload+'     unsigned long desc=')
for i in range(4):reload=reload.replace(f'q[d][{i}]',f'q[{i}]')
reload=reload.replace("'.build-pool3'", "'.build-reload3'").replace("name='codex_verify_pool3'", "name='codex_verify_reload3'")
(p/'reload3.py').write_text(reload)
# Register changes belong AFTER the common prologue barrier, in disjoint branches.
# Earlier variants reconverged with different register limits; inspect ptxas/SASS.
late=src.replace('''if(wg>=2) {
   asm volatile("setmaxnreg.dec.sync.aligned.u32 24;":::"memory");
 }else{
   asm volatile("setmaxnreg.inc.sync.aligned.u32 72;":::"memory");
 }
 ''','')
late=late.replace(' if(wg>=2) {\n   // Several', ' if(wg>=2) {\n   asm volatile("setmaxnreg.dec.sync.aligned.u32 24;":::"memory");\n   // Several')
late=late.replace(' wmma::fragment<wmma::accumulator,16,16,16,float> acc[4];', ' asm volatile("setmaxnreg.inc.sync.aligned.u32 72;":::"memory");\n wmma::fragment<wmma::accumulator,16,16,16,float> acc[4];')
late=late.replace("'.build-dynamic'", "'.build-late3'").replace("name='codex_verify_regpool'", "name='codex_verify_late3'")
(p/'late3.py').write_text(late)
late2=late.replace('__launch_bounds__(384,3)','__launch_bounds__(384,2)').replace('dec.sync.aligned.u32 24','dec.sync.aligned.u32 48').replace('inc.sync.aligned.u32 72','inc.sync.aligned.u32 96').replace("'.build-late3'", "'.build-late2'").replace("name='codex_verify_late3'", "name='codex_verify_late2'").replace('396//','264//')
(p/'late2.py').write_text(late2)

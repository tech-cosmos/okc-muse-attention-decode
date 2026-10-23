from pathlib import Path
s=Path('experiments/gather/dual.py').read_text()
s=s.replace('half V[2][N*(D+8)]','half V[2][N*D]')
s=s.replace('unsigned target=(i/D)*(D+8)+i%D;', 'unsigned vr=i/D,vc=i%D; unsigned target=(vr/8)*512+(vc/8)*64+(vr%8)*8+vc%8;')
# Both head V matrices are overwritten only after asynchronous reads are complete.
s=s.replace('   consumer_barrier(wg);\n   wmma::fragment', '   asm volatile("fence.proxy.async.shared::cta;":::"memory");\n   consumer_barrier(wg);\n   wmma::fragment')
r=Path('experiments/gather/wgmma_pv_reference.py').read_text()
a=r.index('   asm volatile("wgmma.fence.sync.aligned;"',r.index('unsigned pr[2][4]'))
b=r.index('   consumer_barrier();',a)
block=r[a:b]
regs=', '.join(f'"+f"(acc[wg][{n}].x[{i}])' for n in range(4) for i in range(8))
fence='   asm volatile("" : '+regs+' :: "memory");\n'
block=block.replace('acc[','acc[wg][').replace('(V+d*1024)','(V[wg]+d*1024)')
block=block.replace('((unsigned long)__cvta_generic_to_shared(V[wg]+d*1024)>>4)', '(((unsigned long)__cvta_generic_to_shared(V[wg]+d*1024)&0x3ffff)>>4)')
a=s.index('   #pragma unroll\n   for(int n=0;n<4;n++) {\n     #pragma unroll\n     for(int d=0;d<2;d++)')
b=s.index('   consumer_barrier(wg);',a)
s=s[:a]+fence+block+fence+s[b:]
s=s.replace("'.build-dual'","'.build-dual-wgmma'").replace('codex_verify_dual3','codex_verify_dual3wg')
Path('experiments/gather/dual_wgmma.py').write_text(s)
s=Path('experiments/gather/bench_dual.py').read_text().replace('import dual as','import dual_wgmma as').replace('results/dual-','results/dual-wgmma-')
Path('experiments/gather/bench_dual_wgmma.py').write_text(s)

from pathlib import Path
s=Path('experiments/gather/pair_tma.py').read_text()
s=s.replace(',const __grid_constant__ CUtensorMap kmap)',')')
s=s.replace('__shared__ __align__(32) half V[2][N*(D+8)];','__shared__ __align__(32) half V[2][N*(D+8)];\n __shared__ __align__(1024) unsigned char transposed_k[N*128];')
a=s.index('         asm volatile("cp.async.bulk.tensor.2d.');b=s.index('\n         asm volatile("cp.async.bulk.shared::cta',a)
s=s[:a]+'''         asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1],128,[%2];"::"r"(smk),"l"(kc+(long)row*HKV*D+blockIdx.z*128),"r"(bar):"memory");'''+s[b:]
s=s.replace('raw[stage][0][i^((rr&7)<<4)]','raw[stage][0][i]')
a=s.index('   #pragma unroll\n   for(unsigned j=0;j<N*D/256;j++) {')
s=s[:a]+'''   unsigned ro=tid/4,dd=(tid%4)*16,off=ro*128+wg*64+dd;
   *(uint4*)(transposed_k+(off^((ro&7)<<4)))=*(const uint4*)(raw[stage][0]+off);
   asm volatile("fence.proxy.async.shared::cta;":::"memory");
'''+s[a:]
s=s.replace('raw[stage][0]+wg*64+d*32','transposed_k+wg*64+d*32')
a=s.index(' CUtensorMap kmap;');b=s.index(' auto stream=',a);s=s[:a]+s[b:]
s=s.replace('pt.size(1),splits,scale,kmap)','pt.size(1),splits,scale)')
s=s.replace("'.build-pair-tma'","'.build-pair-bulk'").replace('codex_verify_pair3tma','codex_verify_pair3bulk')
Path('experiments/gather/pair_bulk.py').write_text(s)
s=Path('experiments/gather/bench_pair_tma.py').read_text().replace('pair_tma','pair_bulk').replace('pair-tma-','pair-bulk-')
Path('experiments/gather/bench_pair_bulk.py').write_text(s)

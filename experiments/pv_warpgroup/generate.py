"""Build paired-head WGMMA PV from the retained vector-staging baseline."""
from pathlib import Path
p=Path(__file__).parent
s=(p/'baseline.py').read_text()
old=(p/'ancestor_v9.py').read_text()
start=old.index('   asm volatile("wgmma.fence.sync.aligned;',old.index('unsigned pr[2][4]'))
end=old.index('   consumer_barrier();',start)
pv=old[start:end]
pv=pv.replace('V+d*1024','V[wg]+d*1024').replace('((unsigned long)__cvta_generic_to_shared(V[wg]+d*1024)>>4)','(((unsigned long)__cvta_generic_to_shared(V[wg]+d*1024)&0x3ffff)>>4)')
constraints=', '.join(f'"+f"(acc[{i//8}].x[{i%8}])' for i in range(32))
compiler_fence='   asm volatile("" : '+constraints+' :: "memory");\n'
pv=compiler_fence+pv+compiler_fence
a=s.index('   #pragma unroll\n   for(int n=0;n<4;n++) {',s.index('if(__any_sync'))
# Scaling inside the conditional is indented further; this matches the PV loop.
b=s.index('   consumer_barrier(wg);',a)
s=s[:a]+pv+s[b:]
s=s.replace('half V[2][N*(D+8)]','half V[2][N*D]')
s=s.replace('unsigned target=(i/D)*(D+8)+i%D;', 'unsigned vr=i/D, vc=i%D;\n     unsigned target=(vr/8)*512+(vc/8)*64+(vr%8)*8+vc%8;')
s=s.replace('   consumer_barrier(wg);\n   wmma::fragment', '   asm volatile("fence.proxy.async.shared::cta;":::"memory");\n   consumer_barrier(wg);\n   wmma::fragment')
def save(name,src):
 src=src.replace('.build-forced','.build-'+name).replace('codex_matrix_forced','codex_pv_'+name)
 (p/(name+'.py')).write_text(src)
save('sync',s)
# Alternate V buffers: issue PV and defer its wait until the next QK wait.
# Raw FP8 storage may be released once conversion is complete. QK wait also
# finishes preceding PV, before rescale/read and reuse of the older V buffer.
t=s.replace('half V[2][N*D]','half V[2][2][N*D]').replace('V[wg]+','V[wg][tile%2]+')
endwait='   asm volatile("wgmma.commit_group.sync.aligned;\\nwgmma.wait_group.sync.aligned 0;":::"memory");\n'+compiler_fence
assert endwait in t
t=t.replace(endwait,'   asm volatile("wgmma.commit_group.sync.aligned;":::"memory");\n',1)
qwait='   asm volatile("wgmma.commit_group.sync.aligned;\\nwgmma.wait_group.sync.aligned 0;":::"memory");'
assert t.count(qwait)==1
t=t.replace(qwait,qwait+'\n'+compiler_fence)
t=t.replace(' long base=((long)', ' asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\n'+compiler_fence+' long base=((long)')
save('overlap',t)
# N-major SW128 layout gives consecutive 128-byte rows with XOR-swizzled
# 16-byte cells. Align the pattern to 1024 bytes so descriptor base-offset=0.
for name,src in [('swizzle',s),('swizzle_overlap',t)]:
 src=src.replace('__shared__ __align__(32) half V','__shared__ __align__(1024) half V')
 src=src.replace('unsigned target=(vr/8)*512+(vc/8)*64+(vr%8)*8+vc%8;',
                 'unsigned target=i ^ (((i>>6)&7)<<3);')
 src=src.replace('(64ull<<16) | (8ull<<32)', '(64ull<<16) | (64ull<<32) | (1ull<<62)')
 save(name,src)
 # At the QK wait, raw K reads have finished and raw V was already converted
 # before the consumer barrier. Publish the raw-ring slot immediately; the
 # independent converted V buffer remains protected by the PV completion wait.
 if name=='swizzle':
  early=src.replace('   consumer_barrier(wg);\n   if(tid==0)mbarrive(empty+stage);','')
  early=early.replace('   bool isnew=split==0 && tile==tiles-1;',
      '   if(tid==0)mbarrive(empty+stage);\n   bool isnew=split==0 && tile==tiles-1;')
  save('early_release',early)

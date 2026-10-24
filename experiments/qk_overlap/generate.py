"""Reproduce only native scheduling changes from session11 rawvk64 baseline."""
from pathlib import Path
p=Path(__file__).resolve().parent
base=(p/'baseline.py').read_text()
start=base.index('   #pragma unroll\n   for(unsigned j=0;j<N*D/1024;j++)')
end=base.index('   asm volatile("fence.proxy.async.shared::cta;',start)
convert=base[start:end]
# Keep the original K proxy fence + consumer convergence barrier before QK.
s=base[:start]+base[end:]
commit='   asm volatile("wgmma.commit_group.sync.aligned;\\nwgmma.wait_group.sync.aligned 0;":::"memory");'
assert s.count(commit)==2
post='   asm volatile("fence.proxy.async.shared::cta;":::"memory");\n   consumer_barrier(wg);\n'
for name,late in [('overlap',False),('latebar',True)]:
 vstage=convert+('   asm volatile("fence.proxy.async.shared::cta;":::"memory");\n' if late else post)
 out=s.replace(commit,'   asm volatile("wgmma.commit_group.sync.aligned;":::"memory");\n'+vstage+'   asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");',1)
 if late:
  marker='   asm volatile("" : "+f"(acc[0].x[0])'
  assert out.count(marker)==2
  at=out.index(marker)
  out=out[:at]+'   consumer_barrier(wg); // Publish converted V before PV, after independent softmax.\n'+out[at:]
 out=out.replace('.build-baseline','.build-'+name).replace('codex_qko_baseline','codex_qko_'+name)
 (p/(name+'.py')).write_text(out)

# Each consumer acquires the ready mbarrier before its K proxy fence. No
# consumer writes K; the extra pre-QK consumer barrier has no publication role.
for source,name in [('overlap','nobar'),('latebar','latebar_nobar')]:
 out=(p/(source+'.py')).read_text()
 before='   consumer_barrier(wg);\n   wmma::fragment<wmma::accumulator,16,16,16,float> score[2];'
 assert out.count(before)==1
 out=out.replace(before,'   wmma::fragment<wmma::accumulator,16,16,16,float> score[2];')
 out=out.replace('.build-'+source,'.build-'+name).replace('codex_qko_'+source,'codex_qko_'+name)
 (p/(name+'.py')).write_text(out)

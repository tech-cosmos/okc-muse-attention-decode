"""Reproduce two legal single-tile schedules from exact production CUDA."""
import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parent
mod=ast.parse(Path('submission.py').read_text())
call=next(n.value for n in mod.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_NATIVE_SLIDING' for t in n.targets))
base=ast.literal_eval(next(k.value for k in call.keywords if k.arg=='cuda_sources'))
binding=ast.literal_eval(next(k.value for k in call.keywords if k.arg=='cpp_sources'))
(ROOT/'base.cu').write_text(base)
(ROOT/'binding.txt').write_text(binding)
start=base.index(' #pragma unroll\n for(int j=0;j<4;j++)')
end=base.index(' float score[32]',start)
convert=base[start:end]
without=base[:start]+base[end:]
# K is written by ordinary stores for the new row: retain proxy fence/barrier
# before QK; the moved conversion keeps its own V proxy fence and CTA barrier.
assert without[:start].endswith(' __syncthreads();\n')
without=without[:start-len(' __syncthreads();\n')]+' asm volatile("fence.proxy.async.shared::cta;":::"memory");\n __syncthreads();\n'+without[start:]
wait=' asm volatile("wgmma.commit_group.sync.aligned;\\nwgmma.wait_group.sync.aligned 0;":::"memory");'
assert without.count(wait)==2
# Asynchronous accumulator remains compiler-pinned by the original post-wait asm.
overlap=without.replace(wait,' asm volatile("wgmma.commit_group.sync.aligned;":::"memory");\n'+convert+' asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");',1)
late=without.replace(' unsigned pr[4][4];',convert+' unsigned pr[4][4];',1)
pin=' asm volatile("" : '+','.join('"+r"(qr[%d][%d])'%(d,i) for d in range(2) for i in range(4))+' :: "memory");'
pinned=overlap.replace(' asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");',' asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\n'+pin,1)
for name,src in [('overlap',overlap),('late',late),('pinned',pinned)]:
 (ROOT/(name+'.cu')).write_text(src)
 (ROOT/(name+'.py')).write_text((ROOT/'base.py').read_text().replace(".build-base",".build-"+name).replace('codex_sliding_base_rows','codex_sliding_qk_'+name).replace("'base.cu'",repr(name+'.cu')))

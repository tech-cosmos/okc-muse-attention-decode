"""Query CUDA's theoretical residency for the compiled, unchanged attention kernels."""
import ctypes
import fcntl
import json
import hashlib
import sys
from pathlib import Path
sys.path.insert(0,'.')
from experiments.native_merge_pdl.compare import MODULES,cases,check,torch,reference
root=Path(__file__).resolve().parent/'results'
cuda=ctypes.CDLL('libcuda.so.1')
occupancy=cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor
occupancy.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.c_void_p,ctypes.c_int,ctypes.c_size_t]
occupancy.restype=ctypes.c_int
records=[]
with open('/tmp/okc-gpu.lock','w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    for index in (3,13):
        data=reference.generate_input(seed=27103,**cases()[index])
        for name,m in MODULES.items():
            old=set(m._attn_split.device_caches[0][0])
            out=m.kernel_fn(data);check(data,out,(index,name,'occupancy setup'))
            cache=m._attn_split.device_caches[0][0]
            new=set(cache)-old;assert len(new)==1,(name,new)
            kernel=cache[new.pop()]
            blocks=ctypes.c_int()
            err=occupancy(ctypes.byref(blocks),ctypes.c_void_p(kernel.function),128,kernel.metadata.shared)
            assert err==0,err
            row=dict(case=index,mode=name,registers=kernel.n_regs,shared_bytes=kernel.metadata.shared,
                     cuda_max_active_ctas=blocks.value,sm_count=torch.cuda.get_device_properties(0).multi_processor_count,
                     attention_ctas=304,hash=kernel.hash)
            sass='\n'.join(line.rstrip() for line in kernel.asm['sass'].splitlines()).rstrip()+'\n'
            (root/f'case{index}-{name}.sass').write_text(sass)
            row['sass_text_sha256']=hashlib.sha256(sass.encode()).hexdigest()
            records.append(row);print(json.dumps(row),flush=True)
(root/'occupancy.json').write_text(json.dumps(records,indent=2)+'\n')
print('PASS 8 independent output/cache checks',flush=True)

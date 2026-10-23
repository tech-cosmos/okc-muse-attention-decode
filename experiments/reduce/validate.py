"""Run full suite or changed-input graph stress for an installed reduction wrapper.
Usage: validate.py full|fresh|sanitize warp|tile8|tile16 W
"""
import sys,fcntl
sys.path[:0]=['.','experiments/reduce']
import torch,reference,submission,native,tiled
from tools import bench
mode,which,w=sys.argv[1],sys.argv[2],int(sys.argv[3])
orig=submission._combine
native_wrapper=native.Wrapper(w) if which in ('warp','production') else tiled.Wrapper(w,int(which[4:]))
class Guard:
    def __getitem__(self,grid):
        def run(*args,**kw):
            # Dispatch scope under test. All full shapes with at most 128 splits.
            if kw['S']<=128:native_wrapper[grid](*args,**kw)
            else:orig[grid](*args,**kw)
        return run
if which!='production':submission._combine=Guard()
with open('/tmp/okc-gpu.lock','w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    if mode=='full':sys.exit(bench.test())
    cases=[dict(batch=b,ctx=ctx,heads=h,layer='full',q_len=l,page=p,kv_scale=sc,dist=di,spread='mixed') for b,l,h,ctx,p,sc,di in [
        (1,1,'tp8',257,1,1,'normal'),(1,8,'tp8',4097,64,1/64,'soft'),
        (2,8,'tp8',1000,1,1,'edges'),(3,1,'tp8',1024,64,1/64,'normal'),
        (1,8,'dp',2000,64,1,'coherent'),(4,8,'tp8',4000,1,1/64,'soft'),
        (1,1,'tp8',80000,1,1,'coherent'),(1,8,'tp8',120000,1,1,'coherent'),
        (12,8,'tp8',8000,64,1,'soft')]]
    if mode=='sanitize':cases=cases[:4]
    for n,c in enumerate(cases):
        data=reference.generate_input(seed=200+n,**c)
        out=submission.kernel_fn(data);torch.cuda.synchronize()
        ok,msg=reference.check_implementation(data,reference.collect(data,out));assert ok,(c,msg)
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):out=submission.kernel_fn(data)
        for i in range(3 if mode=='sanitize' else 8):
            fresh=reference.generate_input(seed=900+i*100+n,**c)
            # Fixed tensor shapes allow copies of all values/routing fields into graph storage.
            for key,val in data.items():
                if isinstance(val,torch.Tensor) and key!='scheduler_metadata':
                    assert fresh[key].shape==val.shape,(key,c,val.shape,fresh[key].shape)
                    val.copy_(fresh[key])
            # Also exercise empty and shorter contexts using valid existing page rows.
            if i%3==2:
                data['seq_lens'].div_(3,rounding_mode='floor');data['seq_lens'][0]=0
                data['positions'].copy_((data['seq_lens'][:,None]+torch.arange(c['q_len'],device='cuda')[None,:]).flatten())
                data['cache_seqlens'].copy_(data['seq_lens']+c['q_len'])
                pos=data['positions'].reshape(c['batch'],c['q_len']);page=c['page']
                cache_rows=data['page_table'].gather(1,pos//page).long()*page+pos%page
                data['cache_loc'].copy_(cache_rows.flatten())
            g.replay();torch.cuda.synchronize()
            ok,msg=reference.check_implementation(data,reference.collect(data,out));assert ok,(c,i,msg)
            del fresh
        print('FRESH PASS',c,flush=True)
        del data,out,g;torch.cuda.empty_cache()

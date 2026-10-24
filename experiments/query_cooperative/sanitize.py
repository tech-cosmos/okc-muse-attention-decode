"""Fresh native graph replays with independently reduced FP32 partials."""
import fcntl,sys
sys.path.insert(0,'.')
import torch,reference
from experiments.query_cooperative.native import module

with open('/tmp/okc-gpu.lock','w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    for page in (1,64):
        for scale in (1.,1./64):
            for ctx in (10,257,4096):
                c=dict(batch=3,ctx=ctx,heads='tp8',layer='sliding',page=page,q_len=1,kv_scale=scale,dist='edges')
                d=reference.generate_input(seed=3301,**c)
                b,h,hq=3,2,16
                splits=min(32,(d['page_table'].shape[1]*page+63)//64)
                pa=torch.empty((b,h,splits,8,64),device='cuda');pm=torch.empty((b,h,splits,8,2),device='cuda')
                inv=1./(d['config']['rope_theta']**(torch.arange(0,64,2,device='cuda',dtype=torch.float32)/64))
                ang=torch.arange(ctx+1,device='cuda')[:,None]*inv
                cs=torch.cat([ang.cos(),ang.sin()],-1).contiguous()
                def run():
                    module.attention(d['qkv'],cs,d['k_cache'],d['v_cache'],d['page_table'],d['seq_lens'],d['cache_loc'],d['k_scale'],d['v_scale'],pa,pm,page,d['config']['sm_scale']*d['config']['qk_scale_factor']*1.4426950408889634,True)
                run();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):run()
                for rep in range(2):
                    new=reference.generate_input(seed=3310+rep,**c)
                    for k,v in d.items():
                        if isinstance(v,torch.Tensor):v.copy_(new[k])
                    if rep==1:
                        d['seq_lens'].zero_();d['positions'].zero_();d['cache_loc'].copy_((d['page_table'][:,0].long()*page).flatten());d['cache_seqlens'].fill_(1)
                    g.replay();torch.cuda.synchronize()
                    assert torch.isfinite(pa).all() and torch.isfinite(pm).all()
                    w=torch.exp2(pm[...,0]-pm[...,0].amax(2,keepdim=True))
                    a=((pa*w[...,None]).sum(2)/(w*pm[...,1]).sum(2)[...,None]*d['v_scale']).reshape(b,1,hq,64)
                    gate=d['qkv'][...,(hq+2*h)*64:].float().reshape_as(a)
                    out=(a*torch.rsqrt(a.square().mean(-1,keepdim=True)+1e-5)*torch.sigmoid(gate)).flatten(2).bfloat16()
                    ok,msg=reference.check_implementation(d,reference.collect(d,out));assert ok,(page,scale,ctx,rep,msg)
                    print('PASS native-only',page,scale,ctx,rep,flush=True)
                del d,new,g,pa,pm;torch.cuda.empty_cache()

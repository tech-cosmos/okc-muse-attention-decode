"""Native-only graph checks; Torch independently reduces FP32 partials."""
import fcntl,sys
sys.path.insert(0,'.')
import torch,triton,reference,candidate

def run():
    count=0
    for bn in (128,256):
        ext=candidate.extension(bn)
        for page in (1,64):
            for scale in (1.,1/64):
                d=reference.generate_input(batch=2,ctx=1025,heads='tp8',layer='full',page=page,q_len=1,kv_scale=scale,seed=28155)
                c=d['config'];s=triton.cdiv(d['page_table'].shape[1]*page,bn)
                pa=torch.empty((2,2,s,8,64),device='cuda');pm=torch.empty((2,2,s,8,2),device='cuda')
                def invoke():
                    ext.attention(d['qkv'],d['k_scale'],d['k_cache'],d['v_cache'],d['page_table'],d['seq_lens'],d['cache_loc'],d['k_scale'],d['v_scale'],pa,pm,page,c['sm_scale']*c['qk_scale_factor']*candidate.parent.LOG2E,True)
                invoke();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):invoke()
                for seed in (28156,28157,28158):
                    new=reference.generate_input(batch=2,ctx=1025,heads='tp8',layer='full',page=page,q_len=1,kv_scale=scale,seed=seed)
                    for k,v in d.items():
                        if isinstance(v,torch.Tensor):v.copy_(new[k])
                    if seed==28158:
                        d['seq_lens'].zero_();d['positions'].zero_();d['cache_loc'].copy_(d['page_table'][:,0].long()*page);d['cache_seqlens'].fill_(1)
                    g.replay();torch.cuda.synchronize()
                    m=pm[...,0].amax(2,keepdim=True);w=torch.exp2(pm[...,0]-m);den=(w*pm[...,1]).sum(2)
                    a=(pa*w[...,None]).sum(2)/den[...,None]*d['v_scale'];a=a.reshape(2,16,64)
                    a*=torch.rsqrt((a*a).mean(-1,keepdim=True)+1e-5)
                    gate=d['qkv'][:,0,(16+4)*64:].float().reshape(2,16,64)
                    out=(a*torch.sigmoid(gate)).to(torch.bfloat16).reshape(2,1,1024)
                    ok,msg=reference.check_implementation(d,reference.collect(d,out));print('PASS' if ok else 'FAIL',bn,page,scale,seed,msg,flush=True);assert ok;count+=1
    print('TOTAL',count,flush=True)
if __name__=='__main__':
    with open('/tmp/okc-gpu.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);run()

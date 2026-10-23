"""Fresh split/weight/gate graph replays against an independent torch reduction."""
import sys,fcntl
sys.path[:0]=['.','experiments/reduce']
import torch,submission

def check_case(b,l,hkv,s,w,replays):
    hq=hkv*8;rows=l*8
    qkv=torch.empty((b,l,(2*hq+2*hkv)*64),device='cuda',dtype=torch.bfloat16)
    pa=torch.empty((b,hkv,s,rows,64),device='cuda')
    pm=torch.empty((b,hkv,s,rows,2),device='cuda')
    vs=torch.empty((),device='cuda')
    out=torch.empty((b,l,hq*64),device='cuda',dtype=torch.bfloat16)
    def fn():submission._NATIVE_MERGE.combine_full(qkv,pa,pm,vs,out,hq,hkv,l,s,False)
    fn();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):fn()
    for i in range(replays):
        qkv.normal_();pa.normal_();pm[...,0].normal_(0,15 if i%2 else 1);pm[...,1].uniform_(.1,100)
        vs.fill_(1 if i%2 else 1/64)
        # Empty splits must neither dominate a real maximum nor contribute values.
        if i%3==0:
            pm[:,:,1::3,:,0]=-1e30;pm[:,:,1::3,:,1]=0;pa[:,:,1::3]=0
        g.replay()
        mx=pm[...,0].amax(2,keepdim=True);weight=(pm[...,0]-mx).exp2()
        a=(pa*weight[...,None]).sum(2)/(weight*pm[...,1]).sum(2)[...,None]*vs
        a=a.reshape(b,hkv,l,8,64).permute(0,2,1,3,4).reshape(b,l,hq,64)
        a=a*torch.rsqrt(a.square().mean(-1,keepdim=True)+1e-5)
        gate=qkv[...,(hq+2*hkv)*64:].float().reshape(b,l,hq,64)
        expected=(a*gate.sigmoid()).reshape_as(out).bfloat16()
        err=(torch.norm(out.float()-expected.float())/torch.norm(expected.float())).item()
        assert torch.isfinite(out).all() and err<.001,(b,l,hkv,s,w,i,err)
    print('PASS',b,l,hkv,s,w,replays,flush=True)
if __name__=='__main__':
    with open('/tmp/okc-gpu.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for b,l,hkv,s in [(1,1,2,1),(1,8,2,17),(2,8,2,125),(3,1,2,128),(1,8,16,127),(2,8,16,33),(1,1,2,128)]:
            for w in [8]:
                if s>128*w:continue
                check_case(b,l,hkv,s,w,6)

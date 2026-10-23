"""Profile standalone native and Triton merges after input creation, under GPU lock.
This measures instruction/resource use, not kernel_fn's PDL-overlapped runtime.
"""
import sys,fcntl
sys.path[:0]=['.']
import torch,triton,submission
with open('/tmp/okc-gpu.lock','w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    b,l,hq,hkv,s=1,8,16,2,126
    qkv=torch.randn((b,l,(2*hq+2*hkv)*64),device='cuda',dtype=torch.bfloat16)
    pa=torch.randn((b,hkv,s,l*8,64),device='cuda')
    pm=torch.randn((b,hkv,s,l*8,2),device='cuda');pm[...,1]=100
    vs=torch.ones((),device='cuda');out=torch.empty((b,l,hq*64),device='cuda',dtype=torch.bfloat16)
    flush=torch.empty(512<<20,device='cuda',dtype=torch.uint8)
    flush.zero_()
    submission._NATIVE_MERGE.combine_full(qkv,pa,pm,vs,out,hq,hkv,l,s,False)
    expected=out.clone()
    flush.zero_()
    submission._combine[(b*l,hq)](qkv,pa,pm,vs,out,HQ=hq,HKV=hkv,G=8,L=l,M=l*8,D=64,S=s,S_PAD=128,SB=128,PDL=False,num_warps=4,launch_pdl=False)
    torch.cuda.synchronize()
    err=(torch.norm(out.float()-expected.float())/torch.norm(expected.float())).item()
    assert err<.001,err
    print('merge relative difference',err)

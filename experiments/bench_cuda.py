"""Tests and cold-L2 CUDA-graph timing for the native experiment.
Run with the repo venv and MAX_JOBS=2. The module targets sm_90a explicitly.
"""
import argparse, sys
sys.path[:0]=['.','tools','experiments']
import torch, reference, submission, bench, cuda_verify

parser=argparse.ArgumentParser()
parser.add_argument('mode',choices=['test','bench','sweep','nearby','profile'],nargs='?',default='bench')
args=parser.parse_args()
orig=submission.kernel_fn

def supported(c):
    return c['q_len']==8 and c['num_q_heads']//c['num_kv_heads']==8 and c['head_dim']==64 and not c['window']

def dispatch(data):
    return cuda_verify.run(data) if supported(data['config']) else orig(data)

if args.mode=='test':
    submission.kernel_fn=dispatch
    sys.exit(bench.test())
elif args.mode=='profile':
    load,c=bench.SCORED[9]
    data=bench.data_for(load,c)
    for _ in range(3): cuda_verify.run(data)
    torch.cuda.synchronize()
else:
    shapes=[(load,c) for load,c in bench.SCORED if c['layer']=='full' and c['q_len']==8]
    if args.mode=='nearby':
        shapes=[(f'verify {b}',dict(batch=b,ctx=ctx,heads='tp8',layer='full',page=p,q_len=8,dist=dist,spread='mixed',kv_scale=scale))
                for b,ctx,p,dist,scale in [(1,10,1,'normal',1),(2,9000,64,'edges',1/64),(4,40000,1,'soft',1),
                (8,40000,1,'coherent',1),(12,80000,1,'coherent',1),(24,120000,1,'coherent',1),(48,80000,1,'coherent',1)]]
    for load,c in shapes:
        data=bench.data_for(load,c)
        s=max(1,528//(c['batch']*2))
        plans=[(s,3)]
        if args.mode=='sweep':
            plans=[(max(1,n//(c['batch']*2)),st) for n in (396,528,660,792) for st in (3,4,5,6,7)]
        out=cuda_verify.run(data,s)
        ok,msg=reference.check_implementation(data,reference.collect(data,out))
        print(load,s,'PASS' if ok else 'FAIL',msg,flush=True)
        if not ok: sys.exit(1)
        baseline=bench.time_graph(lambda:orig(data))-bench.empty_cost()
        for splits,stages in plans:
            if args.mode=='sweep':
                probe=cuda_verify.run(data,splits,stages)
                torch.testing.assert_close(probe,out,rtol=.02,atol=.002)
            t=bench.time_graph(lambda:cuda_verify.run(data,splits,stages))-bench.empty_cost()
            print(load,'splits',splits,'stages',stages,'native_us',round(t,3),'triton_us',round(baseline,3),flush=True)

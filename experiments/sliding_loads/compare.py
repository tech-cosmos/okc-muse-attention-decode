import argparse,fcntl,json,statistics,sys
from pathlib import Path
sys.path.insert(0,'.')
import torch,submission,reference
from experiments.sliding_loads import candidate
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--control',choices=['native','production'],default='native');parser.add_argument('--seed',type=int,default=103);parser.add_argument('--rounds',type=int,default=60);parser.add_argument('--suffix',default='');args=parser.parse_args()
cases=[]
def add(b,ctx=80000,page=1,heads='tp8',scale=1.,dist='soft',spread='agentic'):
    cases.append(dict(batch=b,ctx=ctx,heads=heads,layer='sliding',page=page,q_len=1,kv_scale=scale,dist=dist,spread=spread))
for b in (1,2,3,8,16,32):add(b)
for b,ctx,page,heads in ((1,10,1,'tp8'),(2,1024,64,'tp8'),(3,4000,1,'tp8'),(1,4000,64,'dp')):add(b,ctx,page,heads)
run_native=candidate.native
results=[]
with open('/tmp/okc-gpu.lock','w') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    flush=torch.empty(512<<20,dtype=torch.uint8,device='cuda')
    def capture(fn):
        for _ in range(3):out=fn()
        torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):out=fn()
        return g,out
    x=torch.zeros(1,device='cuda');eg,_=capture(lambda:x.add_(1))
    def time(g):
        flush.zero_();a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True)
        a.record();g.replay();z.record();z.synchronize();return a.elapsed_time(z)*1000
    for i,c in enumerate(cases):
        data=reference.generate_input(seed=args.seed,**c)
        graphs=[];outputs=[]
        control=(lambda d:candidate.native(d,mod=submission._NATIVE_SLIDING)) if args.control=='native' else submission.kernel_fn
        for f in (lambda:control(data),lambda:run_native(data)):
            out=f();torch.cuda.synchronize();ok,msg=reference.check_implementation(data,reference.collect(data,out));assert ok,(c,msg)
            g,out=capture(f);g.replay();torch.cuda.synchronize();ok,msg=reference.check_implementation(data,reference.collect(data,out));assert ok,(c,msg)
            graphs.append(g);outputs.append(out)
        raw=[]
        for pair in range(args.rounds):
            order=[0,1] if pair%2==0 else [1,0]
            vals={j:time(graphs[j]) for j in order};raw.append([vals[0],vals[1],time(eg)])
        for g,out in zip(graphs,outputs):
            g.replay();torch.cuda.synchronize();ok,msg=reference.check_implementation(data,reference.collect(data,out));assert ok,(c,msg)
        empty=statistics.median(r[2] for r in raw)
        rec={'shape':c,'parent_us':statistics.median(r[0] for r in raw)-empty,'native_us':statistics.median(r[1] for r in raw)-empty,'paired_delta_us':statistics.median(r[1]-r[0] for r in raw),'raw':raw};results.append(rec)
        print(json.dumps({k:v for k,v in rec.items() if k!='raw'}),flush=True)
        del graphs,outputs,g,out,data;torch.cuda.empty_cache()
        (ROOT/f'results/{args.control}-{candidate.variant}{args.suffix}.json').write_text(json.dumps(results,indent=2)+'\n')

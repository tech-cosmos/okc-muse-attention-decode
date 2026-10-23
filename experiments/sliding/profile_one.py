import sys,torch
sys.path.insert(0,'.')
from tools import bench
from experiments.sliding.cuda_sliding import run
c=dict(batch=32,ctx=80000,heads='tp8',layer='sliding',page=1,q_len=1,spread='agentic',dist='soft')
data=bench.data_for('decode 32',c)
for i in range(3):run(data,splits=8,warps=4,stages=2,fused=False)
torch.cuda.synchronize()

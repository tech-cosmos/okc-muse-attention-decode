import sys; sys.path.insert(0,'/home/daytona/repo/tools'); sys.path.insert(0,'/home/daytona/repo')
import torch, bench
x=torch.zeros(1,device='cuda'); y=torch.zeros(1,device='cuda')
print('one add', bench.time_graph(lambda: x.add_(1), reps=200))
print('two adds', bench.time_graph(lambda: (x.add_(1), y.add_(1)), reps=200))
print('empty-ish (no kernel)', bench.time_graph(lambda: None, reps=200))

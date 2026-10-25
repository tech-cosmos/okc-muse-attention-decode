"""s101: CUPTI kernel durations (torch.profiler) of one scored shape's graph replays after an L2 flush."""
import sys; sys.path.insert(0, "tools"); sys.path.insert(0, ".")
import torch, bench, submission, collections
from torch.profiler import profile, ProfilerActivity
load, c = [x for x in bench.SCORED if x[0] == sys.argv[1] and x[1]["layer"] == sys.argv[2]][0]
data = bench.data_for(load, c)
fl = torch.empty(512 << 20, dtype=torch.uint8, device="cuda"); sink = torch.zeros((), dtype=torch.int64, device="cuda")
fn = lambda: submission.kernel_fn(data)
fn(); torch.cuda.synchronize()
g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    fn()
    with torch.cuda.graph(g, stream=s): fn()
torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as p:
    for _ in range(20):
        torch.sum(fl.view(torch.int64), 0, out=sink); g.replay()
    torch.cuda.synchronize()
ev = [e for e in p.events() if e.device_type.name == "CUDA"]
d = collections.defaultdict(list); starts = collections.defaultdict(list)
for e in ev: d[e.name[:40]].append(e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time)
for k, v in d.items():
    v = sorted(v); print(f"{k:40s} n {len(v)} median {v[len(v)//2]:.2f} us")
ks = sorted([(e.time_range.start, e.time_range.end, e.name[:24]) for e in ev])
for a, b, n in ks[-8:]: print(f"{a - ks[-8][0]:9.2f} {b - ks[-8][0]:9.2f} {n}")

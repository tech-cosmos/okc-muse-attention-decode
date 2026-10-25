"""s120: vtk-like grid (256 CTAs, 16-CTA clusters) placement: SMs used and max CTAs per SM."""
import sys, collections, torch
sys.argv = ["x"]
src = open("/home/daytona/repo/tools/s120/place.py").read().split("for B in")[0]
exec(src)
o = torch.zeros(256, dtype=torch.int32, device="cuda")
hist = collections.Counter()
for rep in range(50):
    m.run(o, 256, 0); torch.cuda.synchronize()
    per = collections.Counter(o.tolist()); hist[(len(per), max(per.values()))] += 1
print("vtk-like grid 256: (SMs used, max CTAs/SM) over 50 runs:", dict(hist))

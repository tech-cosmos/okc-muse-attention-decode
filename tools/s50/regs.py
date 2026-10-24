"""Registers/spills of _attn_split for a scored shape under plan overrides: regs.py "decode 1 sliding" '{"fuse": true}'"""
import json, sys, torch
sys.path.insert(0, "."); sys.path.insert(0, "tools")
import bench, submission
load, layer = sys.argv[1].rsplit(" ", 1)
c = dict(next(c for l, c in bench.SCORED if l == load and c["layer"] == layer))
plan = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
data = bench.data_for(load, c)
orig = submission._plan
submission._plan = lambda cfg, mp, dev: {**orig(cfg, mp, dev), **plan}
submission.kernel_fn(data); torch.cuda.synchronize()
for k in (submission._attn_split,):
    for dev, cache in k.device_caches.items():
        for key, kern in cache[0].items():
            print(kern.name, "regs", kern.n_regs, "spills", kern.n_spills, "smem", kern.metadata.shared)

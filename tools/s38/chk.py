"""Correctness of a plan override on scored shapes and a few tests: chk.py '<json plan>' [shape ...]"""
import json, sys, torch
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench, reference, submission
P = json.loads(sys.argv[1]); orig = submission._plan
submission._plan = lambda c, m, d: {**orig(c, m, d), **P}
shapes = sys.argv[2:] or ["verify 1 full"]
for ld, c in bench.SCORED:
    if f"{ld} {c['layer']}" not in shapes: continue
    for seed in (1, 2):
        data = reference.generate_input(seed=seed, **c)
        out = submission.kernel_fn(data); torch.cuda.synchronize()
        ok, msg = reference.check_implementation(data, reference.collect(data, out))
        print(ld, c["layer"], seed, "PASS" if ok else "FAIL", msg[:100], flush=True)

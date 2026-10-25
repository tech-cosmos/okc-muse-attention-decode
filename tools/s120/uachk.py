"""s120: exp/hdiag5's d1ua variants (plan ua + sep) vs reference on decode 1 full, via a patched _plan (diag off)."""
import sys; sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench, reference, submission, torch
submission._DIAG = {}
orig = submission._plan
load, c = [x for x in bench.SCORED if f"{x[0]} {x[1]['layer']}" == "decode 1 full"][0]
for seed in (1, 2, 99):
    outs = []
    for ov in ({}, {"splits": 505, "sep": 1, "ua": True}, {"splits": 511, "sep": 1, "ua": True}):
        submission._plan = lambda *a, **k: {**orig(*a, **k), **ov}
        data = reference.generate_input(seed=seed, **c)
        out = submission.kernel_fn(data); torch.cuda.synchronize(); outs.append(out.float().clone())
        ok, msg = reference.check_implementation(data, reference.collect(data, out))
        print(seed, ov, ok, str(msg)[:70], "max|d|", (out.float() - outs[0]).abs().max().item(), flush=True)

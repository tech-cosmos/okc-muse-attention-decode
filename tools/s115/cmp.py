"""Compare kernel_fn outputs under plan overrides on a scored shape (2 seeds). python tools/s113/cmp.py 'verify 1' full '{}' '{"fix":101}'"""
import sys, json, torch; sys.path.insert(0, "tools"); sys.path.insert(0, ".")
import submission, bench, reference
load, layer = sys.argv[1], sys.argv[2]
_, c = [x for x in bench.SCORED if x[0] == load and x[1]["layer"] == layer][0]
plans = [json.loads(p) for p in sys.argv[3:]]
orig = submission._plan
for seed in (99, 5):
    d = reference.generate_input(seed=seed, **c)
    outs = []
    for p in plans:
        submission._plan = lambda cfg, mp, dev, p=p: {**orig(cfg, mp, dev), **p}
        outs.append(submission.kernel_fn(d).float()); torch.cuda.synchronize()
    submission._plan = orig
    for p, o in zip(plans[1:], outs[1:]):
        print(json.dumps(p), "seed", seed, "max abs", (o - outs[0]).abs().max().item(), "rel", ((o - outs[0]).norm() / outs[0].norm()).item())

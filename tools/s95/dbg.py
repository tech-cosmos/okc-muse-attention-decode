import sys; sys.path.insert(0, "tools"); sys.path.insert(0, ".")
import torch, submission, reference
orig = submission._plan
for ctx in (300, 1000, 3000, 20000):
    data = reference.generate_input(seed=5, batch=1, ctx=ctx, heads="tp8", layer="full", page=1, q_len=1, dist="coherent")
    outs = []
    for extra in ({}, {"red": True}):
        submission._plan = (lambda cfg, mp, dev: {**orig(cfg, mp, dev), **extra})
        d2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in data.items()}
        outs.append(submission.kernel_fn(d2).float().view(16, 64)); torch.cuda.synchronize()
    print(ctx, data["seq_lens"].tolist(), data["page_table"].shape, (outs[1] - outs[0]).abs().amax(1).tolist()[:16])
data = reference.generate_input(seed=5, batch=1, ctx=300, heads="tp8", layer="full", page=1, q_len=1, dist="coherent")
for extra in ({}, {"red": True}):
    submission._plan = (lambda cfg, mp, dev: {**orig(cfg, mp, dev), **extra})
    d2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in data.items()}
    o = submission.kernel_fn(d2); torch.cuda.synchronize()
    print(extra, reference.check_implementation(d2, reference.collect(d2, o)), submission._COUNTERS[(o.device, "r")][:6].tolist())
    if extra:
        pml = None

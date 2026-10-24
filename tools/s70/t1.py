import sys; sys.path.insert(0, "."); sys.path.insert(0, "tools")
import torch, reference, submission, bench
for q in (1, 8):
    for ctx, page in ((80000, 1), (10, 64), (6000, 1), (2047, 64)):
        c = dict(batch=1, ctx=ctx, heads="tp8", layer="sliding", page=page, q_len=q, dist="soft")
        data = reference.generate_input(seed=5, **c)
        cfg = data["config"]
        print(q, ctx, page, submission._plan(cfg, data["page_table"].shape[1], "cuda").get("cl"), flush=True)
        out = submission.kernel_fn(data); torch.cuda.synchronize(); r = reference.check_implementation(data, reference.collect(data, out))
        print(r, flush=True)

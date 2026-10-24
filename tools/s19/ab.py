"""Paired A/B of USE_GDEC on decode full layers: H100 and tools/emu.py H200 emulation (splits as planned for 92 SMs
is not rescaled by the plans, so both get the same programs).  python tools/s19/ab.py [batches] [rounds]"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import torch, bench, emu, reference, submission  # noqa: E402
batches = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "16,32").split(",")]
rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 2
blocks = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "0,40").split(",")]
x = torch.zeros(1, device="cuda")
for blk in blocks:
    empty = emu.time_blocked(lambda: x.add_(1), blk, reps=60)
    for batch in batches:
        c = dict(batch=batch, ctx=80000, heads="tp8", layer="full", page=1, q_len=1, dist="coherent",
                 **({"spread": "agentic"} if batch > 1 else {}))
        data = reference.generate_input(seed=99, **c)
        res = {True: [], False: []}
        f = (132 - blk) / 132
        orig = submission._plan.__wrapped__ if hasattr(submission._plan, "__wrapped__") else submission._plan

        def scaled(cfg, mp, dev, orig=orig, f=f):
            p = dict(orig(cfg, mp, dev))
            if "splits" in p and not p.get("gluon"):
                p["splits"] = max(2, round((p["splits"] + 1) * f) - 1) if p.get("gdec") else max(2, round(p["splits"] * f))
            return p
        scaled.__wrapped__ = orig
        submission._plan = scaled
        for _ in range(rounds):
            for g in (False, True):
                submission.USE_GDEC = g
                res[g].append(emu.time_blocked(lambda: submission.kernel_fn(data), blk, 20) - empty)
        print(f"sms {132 - blk} B={batch:2d} triton {' '.join(f'{t:7.2f}' for t in res[False])}  "
              f"gluon {' '.join(f'{t:7.2f}' for t in res[True])}", flush=True)
        del data
        torch.cuda.empty_cache()

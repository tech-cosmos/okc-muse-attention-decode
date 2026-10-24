"""Paired A/B of plan overrides on decode full: H100 real, all-L2 reads, and emulated H200 (splits scaled).
  python tools/s19/ab2.py '<json list of overrides>' [batches=16,32] [rounds=2]"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import torch, emu, reference, submission  # noqa: E402
variants = json.loads(sys.argv[1])
batches = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "16,32").split(",")]
rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 2
orig = submission._plan
x = torch.zeros(1, device="cuda")
for blk, l2 in ((0, False), (0, True), (40, False)):
    f = (132 - blk) / 132
    empty = emu.time_blocked(lambda: x.add_(1), blk, reps=60)
    for batch in batches:
        c = dict(batch=batch, ctx=80000, heads="tp8", layer="full", page=1, q_len=1, dist="coherent",
                 **({"spread": "agentic"} if batch > 1 else {}))
        data = reference.generate_input(seed=99, **c)
        if l2:
            data["page_table"] = data["page_table"] % 4096
        res = [[] for _ in variants]
        for _ in range(rounds):
            for k, v in enumerate(variants):
                def plan(cfg, mp, dev, v=v):
                    p = {**orig(cfg, mp, dev), **v}
                    if blk and "splits" in p:
                        p["splits"] = max(2, round((p["splits"] + 1) * f) - 1) if p.get("gdec") else \
                            max(2, round(p["splits"] * f))
                    return p
                submission._plan = plan
                res[k].append(emu.time_blocked(lambda: submission.kernel_fn(data), blk, 20) - empty)
        submission._plan = orig
        tag = f"sms {132 - blk}{' L2' if l2 else '   '} B={batch:2d}"
        print(tag + "  " + "  |  ".join(f"{json.dumps(v)} {' '.join(f'{t:7.2f}' for t in r)}"
                                        for v, r in zip(variants, res)), flush=True)
        del data
        torch.cuda.empty_cache()

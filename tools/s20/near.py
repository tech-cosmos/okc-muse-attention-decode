"""Paired A/B on decode sliding: main's Triton plan vs the Gluon decode path (gdec, sep) with
splits = clamp(pow2(SP / pairs), lo, 64). Under tools/emu.py (block SMs parked) when block > 0.
python tools/s20/near.py [batches] [rounds] [blocks] [SP] ['<extra plan json>']"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import torch, emu, reference, submission, triton  # noqa: E402
batches = [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else "2,4,8,16,24,32,48,64").split(",")]
rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
blocks = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "0").split(",")]
SP = int(sys.argv[4]) if len(sys.argv) > 4 else 512
extra = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
orig = submission._plan
x = torch.zeros(1, device="cuda")
for blk in blocks:
    submission._SMS[0] = 132 - blk
    empty = emu.time_blocked(lambda: x.add_(1), blk, reps=60)
    for batch in batches:
        c = dict(batch=batch, ctx=80000, heads="tp8", layer="sliding", page=1, q_len=1, dist="soft",
                 **({"spread": "agentic"} if batch > 1 else {}))
        data = reference.generate_input(seed=99, **c)
        pairs = batch
        sp = max(8, min(64, triton.next_power_of_2(max(1, SP // pairs))))
        g = {"gdec": True, "block_n": 32, "stages": 2, "sep": True, "splits": sp, "pdt": "bf16", **extra}
        res = {0: [], 1: []}
        for _ in range(rounds):
            for k in (0, 1):
                submission._plan = (lambda cfg, mp, dev: {**orig(cfg, mp, dev), **g}) if k else orig
                res[k].append(emu.time_blocked(lambda: submission.kernel_fn(data), blk, 20) - empty)
        submission._plan = orig
        med = lambda v: sorted(v)[len(v) // 2]
        print(f"sms {132 - blk} B={batch:2d} S={sp:2d} triton {med(res[0]):7.2f}  gdec {med(res[1]):7.2f}  "
              f"delta {med(res[1]) - med(res[0]):+6.2f}", flush=True)
        del data
        torch.cuda.empty_cache()

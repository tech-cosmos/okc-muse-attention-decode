"""Paired A/B of USE_GDEC (Gluon decode kernel on both decode layers) per scored decode shape, H100 and
tools/emu.py H200 emulation.  python tools/s20/ab.py [filter=sliding] [rounds=3] [blocks=0,40]"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import torch, bench, emu, submission  # noqa: E402
filt = sys.argv[1] if len(sys.argv) > 1 else "sliding"
rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 3
blocks = [int(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "0,40").split(",")]
x = torch.zeros(1, device="cuda")
for blk in blocks:
    submission._SMS[0] = 132 - blk
    empty = emu.time_blocked(lambda: x.add_(1), blk, reps=60)
    for load, c in bench.SCORED:
        if c["q_len"] != 1 or filt not in f"{load} {c['layer']}":
            continue
        data = bench.data_for(load, c)
        res = {False: [], True: []}
        for _ in range(rounds):
            for g in (False, True):
                submission.USE_GDEC = g
                res[g].append(emu.time_blocked(lambda: submission.kernel_fn(data), blk, 20) - empty)
        med = lambda v: sorted(v)[len(v) // 2]
        print(f"sms {132 - blk} {load:10s} {c['layer']:8s} off {med(res[False]):7.2f} on {med(res[True]):7.2f} "
              f"delta {med(res[True]) - med(res[False]):+6.2f}", flush=True)

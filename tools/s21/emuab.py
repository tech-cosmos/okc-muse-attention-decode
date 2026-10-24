"""Paired A/B of plan overrides on verify 16/32 full under tools/emu.py (BLOCK SMs parked; Gluon verify plans size
splits from _num_sms, so both sides plan for 132 - BLOCK SMs). python tools/s21/emuab.py '<A>' '<B>' [rounds] [block]"""
import json, os, sys
os.environ.setdefault("CUDA_MODULE_LOADING", "EAGER")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import torch, bench, emu, submission  # noqa: E402
A, B = json.loads(sys.argv[1]), json.loads(sys.argv[2])
rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
blk = int(sys.argv[4]) if len(sys.argv) > 4 else 40
orig = submission._plan
submission._SMS[0] = 132 - blk
x = torch.zeros(1, device="cuda")
empty = emu.time_blocked(lambda: x.add_(1), blk, reps=60)
for load, c in bench.SCORED:
    if c["layer"] != "full" or "verify" not in load or c["batch"] == 1:
        continue
    data = bench.data_for(load, c)
    res = ([], [])
    for _ in range(rounds):
        for k, plan in enumerate((A, B)):
            submission._plan = lambda cfg, mp, dev, plan=plan: {**orig(cfg, mp, dev), **plan}
            submission.kernel_fn(data)
            torch.cuda.synchronize()
            res[k].append(emu.time_blocked(lambda: submission.kernel_fn(data), blk, 20) - empty)
    a, b = sorted(res[0])[rounds // 2], sorted(res[1])[rounds // 2]
    print(f"sms {132 - blk} {load:10s} A {a:8.2f}  B {b:8.2f}  d {b - a:+7.2f}", flush=True)

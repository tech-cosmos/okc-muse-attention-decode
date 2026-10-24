"""Verify full at nearby batches/contexts: correctness with gbal (batch <= 24) and paired time gbal off/on.
python tools/s21/near.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import torch, bench, reference, submission  # noqa: E402
orig = submission._plan
for batch in (6, 8, 12, 16, 20, 24, 32, 40, 48, 64):
    for ctx in ((40000, 120000) if batch in (8, 24) else (80000,)):
        c = dict(batch=batch, ctx=ctx, heads="tp8", layer="full", page=1, q_len=8, dist="coherent", spread="agentic")
        data = reference.generate_input(seed=7, **c)
        res = []
        for on in (False, True):
            submission._plan = lambda cfg, mp, dev, on=on: {**orig(cfg, mp, dev), "gbal": on}
            if on and batch <= 24:
                out = submission.kernel_fn(data)
                ok, msg = reference.check_implementation(data, reference.collect(data, out))
            res.append(bench.time_graph(lambda: submission.kernel_fn(data)) - bench.empty_cost())
        submission._plan = orig
        print(f"B={batch:2d} ctx={ctx:6d} off {res[0]:8.2f}  gbal {res[1]:8.2f}  d {res[1] - res[0]:+7.2f}  "
              f"{'' if batch > 24 else ('PASS' if ok else 'FAIL ' + msg[:80])}", flush=True)
        del data
        torch.cuda.empty_cache()

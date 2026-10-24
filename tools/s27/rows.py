"""Check and time the row-split sliding verify kernel (plan key rows) against the default plan.
usage: python tools/s27/rows.py check '<json override>' | time '<json list of overrides>' [shape filter]"""
import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
import bench, submission, reference

mode, arg = sys.argv[1], json.loads(sys.argv[2])
flt = sys.argv[3] if len(sys.argv) > 3 else "sliding"
if mode == "check":
    orig = submission._plan
    submission._plan = lambda cfg, mp, dev: ({**orig(cfg, mp, dev), **arg} if cfg["window"] and cfg["q_len"] > 1
                                             else orig(cfg, mp, dev))
    cases = [c for c in bench.TESTS if c.get("layer") == "sliding" and c.get("q_len", 1) > 1]
    cases += [c for load, c in bench.SCORED if c["layer"] == "sliding" and c["q_len"] > 1]
    cases += [dict(batch=b, ctx=ctx, heads="tp8", layer="sliding", q_len=8, page=p, spread="agentic")
              for b, ctx, p in ((7, 1500, 1), (16, 2100, 16), (3, 30, 1), (32, 80000, 64))]
    for i, c in enumerate(cases):
        data = reference.generate_input(seed=777 + i, **c)
        out = submission.kernel_fn(data)
        torch.cuda.synchronize()
        ok, msg = reference.check_implementation(data, reference.collect(data, out))
        print("PASS" if ok else "FAIL", c, msg[:80], flush=True)
else:
    for load, c in bench.SCORED:
        name = f"{load} {c['layer']}"
        if flt not in name or c["q_len"] == 1 or c["layer"] != "sliding":
            continue
        for p in [{}] + arg:
            print(f"{name:20s} {bench.run_shape(load, c, p):7.2f} us  {p}", flush=True)

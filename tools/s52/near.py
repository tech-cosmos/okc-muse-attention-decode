"""Decode full at nearby batches: correctness of a plan override and paired time (real and all-L2), 3 rounds.
python tools/s52/near.py '<json plan>' [batches]"""
import json, os, statistics, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import torch, bench, reference, submission  # noqa: E402
P = json.loads(sys.argv[1]); orig = submission._plan
batches = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [4, 8, 12, 16, 24, 32, 48, 64]
for batch in batches:
    c = dict(batch=batch, ctx=80000, heads="tp8", layer="full", page=1, dist="coherent", spread="agentic")
    data = reference.generate_input(seed=7, **c)
    submission._plan = lambda cfg, mp, dev: {**orig(cfg, mp, dev), **P}
    out = submission.kernel_fn(data)
    ok, msg = reference.check_implementation(data, reference.collect(data, out))
    line = f"B={batch:2d} {'PASS' if ok else 'FAIL ' + msg[:80]}"
    for mode in ("real", "L2"):
        d2 = dict(data)
        if mode == "L2":
            d2["page_table"] = d2["page_table"] % 4096
        res = [[], []]
        for r in range(3):
            for on in ((0, 1) if r % 2 == 0 else (1, 0)):
                submission._plan = (lambda cfg, mp, dev: {**orig(cfg, mp, dev), **P}) if on else orig
                res[on].append(bench.time_graph(lambda: submission.kernel_fn(d2)) - bench.empty_cost())
        d = [b - a for a, b in zip(*res)]
        line += f"  {mode} {statistics.median(res[0]):7.2f} -> {statistics.median(res[1]):7.2f} ({statistics.median(d):+.2f})"
    submission._plan = orig
    print(line, flush=True)
    del data, d2
    torch.cuda.empty_cache()

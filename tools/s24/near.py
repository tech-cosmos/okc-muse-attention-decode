"""Decode full at nearby batches: paired time of plan override A vs B (merge settings), plus correctness of B.
python tools/s24/near.py '<json A>' '<json B>' [rounds=3] [layer=full]"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import torch, bench, reference, submission  # noqa: E402
A, B = json.loads(sys.argv[1]), json.loads(sys.argv[2])
rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
layer = sys.argv[4] if len(sys.argv) > 4 else "full"
orig = submission._plan
for batch in (2, 4, 8, 12, 16, 24, 32, 48, 64):
    c = dict(batch=batch, ctx=80000, heads="tp8", layer=layer, page=1, q_len=1,
             dist="coherent" if layer == "full" else "soft", spread="agentic")
    data = reference.generate_input(seed=7, **c)
    ts = {0: [], 1: []}
    for r in range(rounds):
        for k, p in ((0, A), (1, B)):
            submission._plan = lambda cfg, mp, dev, p=p: {**orig(cfg, mp, dev), **p}
            if r == 0 and k == 1:
                out = submission.kernel_fn(data)
                ok, msg = reference.check_implementation(data, reference.collect(data, out))
            ts[k].append(bench.time_graph(lambda: submission.kernel_fn(data)) - bench.empty_cost())
    submission._plan = orig
    a, b = sorted(ts[0])[rounds // 2], sorted(ts[1])[rounds // 2]
    print(f"B={batch:2d} A {a:8.2f}  B {b:8.2f}  d {b - a:+7.2f}  {'PASS' if ok else 'FAIL ' + msg[:80]}", flush=True)
    del data
    torch.cuda.empty_cache()

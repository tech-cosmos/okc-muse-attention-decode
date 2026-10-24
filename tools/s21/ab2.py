"""Paired A/B of plan overrides at any verify-full shape, real and all-L2.
python tools/s21/ab2.py batch ctx '<A>' '<B>' [rounds]"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import bench, reference, submission  # noqa: E402
batch, ctx = int(sys.argv[1]), int(sys.argv[2])
A, B = json.loads(sys.argv[3]), json.loads(sys.argv[4])
rounds = int(sys.argv[5]) if len(sys.argv) > 5 else 3
orig = submission._plan
c = dict(batch=batch, ctx=ctx, heads="tp8", layer="full", page=1, q_len=8, dist="coherent", spread="agentic")
base = reference.generate_input(seed=99, **c)
for l2 in (False, True):
    data = dict(base)
    if l2:
        data["page_table"] = data["page_table"] % 4096
    res = ([], [])
    for _ in range(rounds):
        for k, plan in enumerate((A, B)):
            submission._plan = lambda cfg, mp, dev, plan=plan: {**orig(cfg, mp, dev), **plan}
            res[k].append(bench.time_graph(lambda: submission.kernel_fn(data)) - bench.empty_cost())
    a, b = sorted(res[0])[rounds // 2], sorted(res[1])[rounds // 2]
    print(f"B={batch} ctx={ctx} {'L2  ' if l2 else 'real'} A {a:8.2f}  B {b:8.2f}  d {b - a:+7.2f}", flush=True)

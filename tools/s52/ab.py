"""Paired plan A/B on decode/verify full, real and all-L2 (page_table % 4096):
python tools/s52/ab.py ROUNDS 'shape,shape' '<json plan>' ['<json plan>' ...]   (plan 0 = {} base)"""
import json, os, statistics, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import bench, submission  # noqa: E402

rounds = int(sys.argv[1]); shapes = sys.argv[2].split(","); plans = [{}] + [json.loads(a) for a in sys.argv[3:]]
orig = submission._plan
for load, c in bench.SCORED:
    name = f"{load} {c['layer']}"
    if name not in shapes:
        continue
    for mode in ("real", "L2"):
        data = dict(bench.data_for(load, c))
        if mode == "L2":
            data["page_table"] = data["page_table"] % 4096
        res = [[] for _ in plans]
        for r in range(rounds):
            order = range(len(plans)) if r % 2 == 0 else reversed(range(len(plans)))
            for i in order:
                submission._plan = lambda cf, mp, dv, P=plans[i]: {**orig(cf, mp, dv), **P}
                res[i].append(bench.time_graph(lambda: submission.kernel_fn(data)) - bench.empty_cost())
            submission._plan = orig
        base = res[0]
        out = [f"{statistics.median(base):7.2f}"]
        for x in res[1:]:
            d = [a - b for a, b in zip(x, base)]
            out.append(f"{statistics.median(x):7.2f} (d {statistics.median(d):+.2f}, {sum(v < 0 for v in d)}/{rounds})")
        print(f"{name:16s} {mode:4s} " + "  ".join(out), flush=True)

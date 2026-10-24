"""Paired A/B of plan overrides on the verify/decode full shapes, real and all-L2 (page_table % 4096).
python tools/s21/ab.py '<json A>' '<json B>' [rounds=3] [filter=verify]"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import bench, submission  # noqa: E402

A, B = json.loads(sys.argv[1]), json.loads(sys.argv[2])
rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
filt = sys.argv[4] if len(sys.argv) > 4 else "verify"
orig = submission._plan


def run(data, plan):
    submission._plan = lambda cfg, mp, dev: {**orig(cfg, mp, dev), **plan}
    try:
        return bench.time_graph(lambda: submission.kernel_fn(data)) - bench.empty_cost()
    finally:
        submission._plan = orig


for load, c in bench.SCORED:
    if c["layer"] != "full" or filt not in load or c["batch"] == 1:
        continue
    for l2 in (False, True):
        data = dict(bench.data_for(load, c))
        if l2:
            data["page_table"] = data["page_table"] % 4096
        ta, tb = [], []
        for _ in range(rounds):
            ta.append(run(data, A))
            tb.append(run(data, B))
        ta.sort(), tb.sort()
        print(f"{load:10s} {'L2  ' if l2 else 'real'} A {ta[len(ta) // 2]:8.2f}  B {tb[len(tb) // 2]:8.2f}  "
              f"d {tb[len(tb) // 2] - ta[len(ta) // 2]:+7.2f}", flush=True)

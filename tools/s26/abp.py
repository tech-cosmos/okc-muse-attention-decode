"""Paired A/B of a plan override on one scored shape (alternating rounds).
python tools/s26/abp.py '<shape>' '<json override>' [rounds=9]"""
import json, os, statistics, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import bench  # noqa: E402

shape, over = sys.argv[1], json.loads(sys.argv[2])
rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 9
for load, c in bench.SCORED:
    if f"{load} {c['layer']}" == shape:
        a, b = [], []
        for _ in range(rounds):
            a.append(bench.run_shape(load, c, {}))
            b.append(bench.run_shape(load, c, over))
        d = [y - x for x, y in zip(a, b)]
        print(shape, over, "A", round(statistics.median(a), 2), "B", round(statistics.median(b), 2),
              "paired median", round(statistics.median(d), 2), "B wins", sum(x < 0 for x in d), "/", rounds)

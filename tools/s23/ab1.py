"""Paired A/B of USE_GD1 on decode 1 full (alternating rounds)."""
import sys, os, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import bench, submission
shape = sys.argv[1] if len(sys.argv) > 1 else "decode 1 full"
for load, c in bench.SCORED:
    if f"{load} {c['layer']}" == shape:
        a, b = [], []
        for _ in range(int(os.environ.get("ROUNDS", 7))):
            submission.USE_GD1 = False; a.append(bench.run_shape(load, c))
            submission.USE_GD1 = True; b.append(bench.run_shape(load, c))
        print(shape, "off", round(statistics.median(a), 2), "on", round(statistics.median(b), 2),
              "diffs", [round(y - x, 2) for x, y in zip(a, b)])

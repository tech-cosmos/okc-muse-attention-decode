"""agent2: paired H100 A/B of plan overrides on one scored shape, alternating order.
  python tools/a2/abp.py "verify 16 sliding" '{}' '{"preq": 1}' [rounds]"""
import sys, json, statistics
sys.path.insert(0, "/home/daytona/repo2/tools"); sys.path.insert(0, "/home/daytona/repo2")
import bench
name, pa, pb = sys.argv[1], json.loads(sys.argv[2]), json.loads(sys.argv[3])
R = int(sys.argv[4]) if len(sys.argv) > 4 else 9
for load, c in bench.SCORED:
    if f"{load} {c['layer']}" != name: continue
    d, A, B = [], [], []
    for r in range(R):
        if r % 2:
            tb = bench.run_shape(load, c, pb); ta = bench.run_shape(load, c, pa)
        else:
            ta = bench.run_shape(load, c, pa); tb = bench.run_shape(load, c, pb)
        A.append(ta); B.append(tb); d.append(tb - ta)
    print(f"{name}: A {statistics.median(A):.2f} B {statistics.median(B):.2f}; paired median B-A {statistics.median(d):+.3f} us, B wins {sum(x < 0 for x in d)}/{R}", flush=True)

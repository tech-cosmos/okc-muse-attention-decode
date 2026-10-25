"""s125: paired plan A/B on scored shapes (cold). usage: ab.py '<shape>' '<planA json>' '<planB json>' [rounds]"""
import sys, json, statistics; sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench
shape = sys.argv[1]; A, B = json.loads(sys.argv[2]), json.loads(sys.argv[3]); rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 9
load, c = [x for x in bench.SCORED if f"{x[0]} {x[1]['layer']}" == shape][0]
a, b = [], []
for r in range(rounds):
    for lst, p in ((a, A), (b, B)) if r % 2 == 0 else ((b, B), (a, A)):
        lst.append(bench.run_shape(load, c, plan=p))
d = [y - x for x, y in zip(a, b)]
print(f"{shape}: A {statistics.median(a):.2f} B {statistics.median(b):.2f} B-A median {statistics.median(d):+.2f} "
      f"({sum(x < 0 for x in d)}/{len(d)} B faster)", flush=True)

import sys, json; sys.path.insert(0, "tools"); sys.path.insert(0, ".")
import bench
load, c = bench.SCORED[int(sys.argv[2]) if len(sys.argv) > 2 else 7]
plans = json.loads(sys.argv[1])
for _ in range(3):
    print(" | ".join("%s %.2f" % (json.dumps(p), bench.run_shape(load, c, p)) for p in plans), flush=True)

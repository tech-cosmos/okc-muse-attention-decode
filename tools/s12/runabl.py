import sys, json; d=sys.argv[1]; sys.path.insert(0, d); sys.path.insert(1, "/home/daytona/repo/tools"); sys.path.insert(2, "/home/daytona/repo")
import submission; assert submission.__file__.startswith(d), submission.__file__
import bench
plans = json.loads(sys.argv[2])
for load in __import__("os").environ.get("LOADS","verify 16").split(","):
    for l, c in bench.SCORED:
        if l == load and c["layer"] == "full":
            for p in plans:
                print(d, load, round(bench.run_shape(l, c, p), 1), p, flush=True)

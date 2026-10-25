"""agent2: time the 4 batched sliding shapes (current plans). python tools/a2/b4.py [reps]"""
import sys, statistics
sys.path.insert(0, "/home/daytona/repo2/tools"); sys.path.insert(0, "/home/daytona/repo2")
import bench
R = int(sys.argv[1]) if len(sys.argv) > 1 else 3
for load, c in bench.SCORED:
    if c["layer"] == "sliding" and load in ("decode 16", "decode 32", "verify 16", "verify 32"):
        t = [bench.run_shape(load, c) for _ in range(R)]
        print(f"{load} sliding: {statistics.median(t):.2f} us ({' '.join(f'{x:.2f}' for x in t)})", flush=True)

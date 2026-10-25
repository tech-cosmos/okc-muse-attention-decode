import sys, statistics, importlib
sys.path.insert(0, "/home/daytona/repo2/tools"); sys.path.insert(0, "/tmp")
import bench
A = importlib.import_module("a2sub_4_1"); B = importlib.import_module("a2sub_4_2")
for l, c in bench.SCORED:
    if c["layer"] == "sliding" and l in sys.argv[1:]:
        d, ta, tb = [], [], []
        for r in range(11):
            order = (A, B) if r % 2 == 0 else (B, A)
            t = {}
            for m in order:
                bench.submission = m; t[m] = bench.run_shape(l, c)
            ta.append(t[A]); tb.append(t[B]); d.append(t[B] - t[A])
        print(f"{l}: LAG1 {statistics.median(ta):.2f} LAG2 {statistics.median(tb):.2f} paired {statistics.median(d):+.3f}, LAG2 wins {sum(x < 0 for x in d)}/11", flush=True)

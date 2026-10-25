"""agent2: paired H100 A/B of a submission module global on one scored shape.
  python tools/a2/abg.py "decode 32 sliding" CSL_MAX_BATCH 16 32 [rounds]"""
import sys, statistics
sys.path.insert(0, "/home/daytona/repo2/tools"); sys.path.insert(0, "/home/daytona/repo2")
import bench, submission
name, var, va, vb = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
R = int(sys.argv[5]) if len(sys.argv) > 5 else 9
def run(load, c, v):
    setattr(submission, var, v); return bench.run_shape(load, c)
for load, c in bench.SCORED:
    if f"{load} {c['layer']}" != name: continue
    d, A, B = [], [], []
    for r in range(R):
        if r % 2: tb = run(load, c, vb); ta = run(load, c, va)
        else: ta = run(load, c, va); tb = run(load, c, vb)
        A.append(ta); B.append(tb); d.append(tb - ta)
    print(f"{name} {var}: A={va} {statistics.median(A):.2f} B={vb} {statistics.median(B):.2f}; paired B-A {statistics.median(d):+.3f} us, B wins {sum(x < 0 for x in d)}/{R}", flush=True)

"""Decode 1 full on the H100: whole vs attention only (merge launch dropped), over plan overrides (argv[1] json list)."""
import sys, json, os; sys.path.insert(0, "tools"); sys.path.insert(0, "."); sys.path.insert(0, os.environ.get("SUBDIR", "."))
import submission; import bench; assert bench.submission is submission; print(submission.__file__)
load, c = [x for x in bench.SCORED if x[0] == sys.argv[2] and x[1]["layer"] == "full"][0] if len(sys.argv) > 2 else bench.SCORED[1]
class Nop:
    def __getitem__(self, g): return lambda *a, **k: None
orig = submission._combine
for p in json.loads(sys.argv[1]):
    r = []
    for _ in range(2):
        submission._combine = orig
        w = bench.run_shape(load, c, p)
        submission._combine = Nop()
        a = bench.run_shape(load, c, p)
        r.append((w, a))
    submission._combine = orig
    print(p, " ".join("whole %.2f attn %.2f" % x for x in r), flush=True)

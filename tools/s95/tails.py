"""Whole vs merge-dropped times for the batched full shapes (H100)."""
import sys; sys.path.insert(0, "tools"); sys.path.insert(0, ".")
import submission, bench
class Nop:
    def __getitem__(self, g): return lambda *a, **k: None
orig = submission._combine_bal
for i in (3, 5, 9, 11):
    load, c = bench.SCORED[i]
    r = []
    for _ in range(2):
        submission._combine_bal = orig; w = bench.run_shape(load, c)
        submission._combine_bal = Nop(); a = bench.run_shape(load, c)
        r.append("whole %.2f nomerge %.2f" % (w, a))
    submission._combine_bal = orig
    print(load, c["layer"], " | ".join(r), flush=True)

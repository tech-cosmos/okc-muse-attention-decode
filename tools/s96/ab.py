"""Paired A/B of two submission.py copies: python tools/s96/ab.py <dirA> <dirB> "<load> <layer>" ..."""
import sys, importlib.util, statistics
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench, reference
def load(d, n):
    spec = importlib.util.spec_from_file_location(n, d + "/submission.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
A, B = load(sys.argv[1], "subA"), load(sys.argv[2], "subB")
for shape in sys.argv[3:]:
    ld, layer = shape.rsplit(" ", 1)
    c = [x for x in bench.SCORED if x[0] == ld and x[1]["layer"] == layer][0][1]
    data = bench.data_for(ld, c)
    ra, rb = [], []
    for r in range(6):
        ra.append(bench.time_graph(lambda: A.kernel_fn(data)) - bench.empty_cost())
        rb.append(bench.time_graph(lambda: B.kernel_fn(data)) - bench.empty_cost())
    print(f"{shape}: A {statistics.median(ra):.2f} B {statistics.median(rb):.2f}  wins B {sum(b < a for a, b in zip(ra, rb))}/6", flush=True)

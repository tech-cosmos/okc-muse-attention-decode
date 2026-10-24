"""Paired A/B timing of two submission.py copies: python tools/s18/ab.py <dirA> <dirB> [rounds]; LOADS="decode 1,..."
(full layers; LAYER=sliding for sliding). Alternates A and B each round; prints per-shape medians."""
import sys, os, importlib.util, statistics
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(1, "/home/daytona/repo")
import bench
mods = []
for i, d in enumerate(sys.argv[1:3]):
    spec = importlib.util.spec_from_file_location(f"sub_ab{i}", os.path.join(d, "submission.py"))
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m); mods.append(m)
rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 5
layer = os.environ.get("LAYER", "full")
for load in os.environ.get("LOADS", "decode 1,verify 1").split(","):
    for l, c in bench.SCORED:
        if l == load and c["layer"] == layer:
            ts = [[], []]
            for r in range(rounds):
                for i, m in enumerate(mods):
                    bench.submission = m
                    ts[i].append(bench.run_shape(l, c))
            a, b = statistics.median(ts[0]), statistics.median(ts[1])
            print(f"{load} {layer}: A {a:.2f}  B {b:.2f}  B-A {b - a:+.2f}", flush=True)

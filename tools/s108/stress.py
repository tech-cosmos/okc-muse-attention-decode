"""s108: REDO-trigger stress for main's fixed-reference paths (LZ 10 verify 16/32 full, fix 1 verify 1 full).
Per seed: every scored shape in judge order (sliding then full, all loads) in one process; each call timed with a sync
(a REDO costs ~ms) and checked against the reference. Args: first_seed n_seeds [shape filter]."""
import sys, time, torch
sys.path.insert(0, "/home/daytona/repo"); sys.path.insert(0, "/home/daytona/repo/tools")
import reference as R, bench, submission
s0, n = int(sys.argv[1]), int(sys.argv[2])
filt = sys.argv[3] if len(sys.argv) > 3 else ""
for seed in range(s0, s0 + n):
    for load, c in bench.SCORED:
        name = f"{load} {c['layer']}"
        if filt not in name:
            continue
        d = R.generate_input(seed=seed, **c)
        out = submission.kernel_fn(d); torch.cuda.synchronize()
        ts = []
        for _ in range(3):
            t = time.perf_counter(); submission.kernel_fn(d); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
        ok, msg = R.check_implementation(d, R.collect(d, out))
        t = min(ts) * 1e6
        flag = "  <-- SLOW" if t > 2000 else ""
        print(f"seed {seed} {name:18s} {t:9.1f} us ok={ok}{flag}" + ("" if ok else f" {msg}"), flush=True)
        del d, out

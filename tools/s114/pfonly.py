import sys, torch
sys.argv = [sys.argv[0]]
exec(open("/home/daytona/repo/tools/s114/sweep.py").read().split("for nt, P in")[0])
P = 132 * 32
for nwin in (32, 128, 512):
    wr = (nrows_pool + nwin - 1) // nwin
    off = torch.zeros(P, nwin + 1, dtype=torch.int32, device="cuda")
    t = T(lambda: m.sweep_run(kc, vc, allrows, off, nwin, wr * 128, 100 + nwin, P, 512, sink))
    print(f"prefetch-only whole pool ({nrows_pool * 256 / 1e6:.0f} MB), {nwin} windows: {t:.1f} us", flush=True)

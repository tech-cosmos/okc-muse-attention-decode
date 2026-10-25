import sys, torch
sys.argv = [sys.argv[0]]
exec(open("/home/daytona/repo/tools/s114/sweep.py").read().split("for nt, P in")[0])
P = 132 * 32
for nwin in (32, 128):
    wr = (nrows_pool + nwin - 1) // nwin
    off = torch.zeros(P, nwin + 1, dtype=torch.int32, device="cuda")
    t = T(lambda: m.sweep_run(kc, vc, allrows, off, nwin, wr * 128, 300 + nwin, P, 512, sink))
    print(f"line-prefetch-only whole pool, {nwin} windows: {t:.1f} us", flush=True)
bounds = torch.linspace(0, N, P + 1, device="cuda").long()
prog = torch.searchsorted(bounds[1:], torch.arange(N, device="cuda"), right=True)
for nwin in (16, 32, 64, 128):
    wr = (nrows_pool + nwin - 1) // nwin
    key = prog * nwin + allrows.long() // wr
    order = torch.argsort(key, stable=True)
    rows_s = allrows[order].contiguous()
    cnt = torch.bincount(key, minlength=P * nwin).view(P, nwin)
    off = torch.zeros(P, nwin + 1, dtype=torch.long, device="cuda")
    off[:, 1:] = torch.cumsum(cnt, 1)
    off = (off + bounds[:-1, None]).int().contiguous()
    res = [T(lambda: m.sweep_run(kc, vc, rows_s, off, nwin, wr * 128, d, P, 512, sink)) for d in (-1, 201, 202, 203)]
    print(f"NWIN {nwin}: no prefetch {res[0]:.1f}, line prefetch D 1/2/3 " + " / ".join(f"{x:.1f}" for x in res[1:]), flush=True)

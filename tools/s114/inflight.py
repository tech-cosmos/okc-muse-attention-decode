import sys, torch
sys.argv = [sys.argv[0]]
exec(open("/home/daytona/repo/tools/s114/sweep.py").read().split("for nt, P in")[0])
for P in (132 * 8, 132 * 16, 132 * 32):
    bounds = torch.linspace(0, N, P + 1, device="cuda").long()
    off1 = torch.stack([bounds[:-1], bounds[1:]], 1).int().contiguous()
    res = [T(lambda: m.sweep_run(kc, vc, allrows, off1, 1, nrows_pool * 128, u * 1000, P, 512, sink)) for u in (4, 8, 16)]
    print(f"P {P} warps: random order, rows in flight per warp 16/32/64: " + " / ".join(f"{x:.1f}" for x in res), flush=True)

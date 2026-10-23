"""Ceiling for the full-layer shape: gather B*ctx random 128-byte K and V rows, sum them."""
import torch, triton, triton.language as tl, statistics
@triton.jit
def g(kc, vc, idx, out, CHUNK: tl.constexpr, BN: tl.constexpr, ST: tl.constexpr):
    pid = tl.program_id(0)
    c = tl.arange(0, 128); n = tl.arange(0, BN)
    acc = tl.zeros([BN, 128], tl.float32)
    for s in tl.range(pid * CHUNK, pid * CHUNK + CHUNK, BN, num_stages=ST):
        row = tl.load(idx + s + n).to(tl.int64) * 128
        acc += tl.load(kc + row[:, None] + c[None, :]).to(tl.float32) + tl.load(vc + row[:, None] + c[None, :]).to(tl.float32)
    tl.store(out + pid * 128 + c, tl.sum(acc, 0))
N = 32 * 80000
for page_rows in (1, 16, 64):  # contiguous rows per page
    kc = torch.randint(0, 255, (N, 128), dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
    vc = torch.randint(0, 255, (N, 128), dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
    perm = torch.randperm(N // page_rows, device="cuda")
    idx = (perm[:, None] * page_rows + torch.arange(page_rows, device="cuda")).flatten().int()
    flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
    best = []
    for chunk in (1024, 2560, 5120):
        for bn, w, st in [(32, 1, 3), (32, 2, 3), (64, 2, 3), (64, 4, 3), (64, 4, 4), (128, 4, 3), (32, 1, 4)]:
            out = torch.empty((N // chunk, 128), device="cuda")
            f = lambda: g[(N // chunk,)](kc, vc, idx, out, CHUNK=chunk, BN=bn, ST=st, num_warps=w)
            f(); torch.cuda.synchronize(); ts = []
            for _ in range(15):
                flush.zero_(); a = torch.cuda.Event(True); b = torch.cuda.Event(True)
                a.record(); f(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1000)
            best.append((statistics.median(ts), chunk, bn, w, st))
    best.sort()
    t = best[0][0]
    print(f"page_rows={page_rows}: best {t:.1f} us = {2 * N * 128 / t / 1e6:.2f} TB/s  {best[:3]}")

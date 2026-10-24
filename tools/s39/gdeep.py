"""Is the random-128B-row gather ceiling (~236 us, 2.78 TB/s on H100) a DRAM limit or a kernel limit?
Deeper in-flight configs of the gather kernel, plus a plain dense read (torch) of the same bytes."""
import torch, triton, triton.language as tl, statistics, fcntl
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
kc = torch.randint(0, 255, (N, 128), dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
vc = torch.randint(0, 255, (N, 128), dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
perm = torch.randperm(N, device="cuda").int()
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
lock = open("/tmp/okc-gpu.lock", "w")
def tm(f, reps=15):
    f(); torch.cuda.synchronize(); ts = []
    fcntl.flock(lock, fcntl.LOCK_EX)
    for _ in range(reps):
        flush.zero_(); a = torch.cuda.Event(True); b = torch.cuda.Event(True)
        a.record(); f(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1000)
    fcntl.flock(lock, fcntl.LOCK_UN)
    return statistics.median(ts)
both = torch.cat([kc.view(torch.uint8).view(-1), vc.view(torch.uint8).view(-1)]).view(torch.int64)
t = tm(lambda: both.sum()); print(f"torch dense int64 sum: {t:.1f} us = {2 * N * 128 / t / 1e6:.2f} TB/s")
t = tm(lambda: both.view(torch.float64).amax()); print(f"torch dense amax: {t:.1f} us = {2 * N * 128 / t / 1e6:.2f} TB/s")
res = []
for chunk in (320, 640, 1280, 2560):
    for bn, w, st in [(16, 1, 6), (16, 1, 8), (32, 1, 6), (32, 2, 6), (32, 1, 8), (64, 2, 6), (64, 4, 8), (128, 8, 4)]:
        if chunk % bn: continue
        out = torch.empty((N // chunk, 128), device="cuda")
        t = tm(lambda: g[(N // chunk,)](kc, vc, perm, out, CHUNK=chunk, BN=bn, ST=st, num_warps=w), 9)
        res.append((round(t, 1), chunk, bn, w, st))
res.sort(); print("random rows best:", [(t, f"{2 * N * 128 / t / 1e6:.2f}TB/s", *r) for t, *r in res[:6]])

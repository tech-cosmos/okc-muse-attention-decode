"""Gather floor for decode 32 sliding: 32 x 2047 random rows of a 2.75M-row cache, 128B (both heads) or 64B halves."""
import torch, triton, triton.language as tl, statistics, fcntl
@triton.jit
def g(kc, vc, idx, out, CHUNK: tl.constexpr, BN: tl.constexpr, ST: tl.constexpr, W: tl.constexpr, H: tl.constexpr):
    pid = tl.program_id(0); hh = tl.program_id(1)
    c = tl.arange(0, W); n = tl.arange(0, BN)
    acc = tl.zeros([BN, W], tl.float32)
    for s in tl.range(pid * CHUNK, pid * CHUNK + CHUNK, BN, num_stages=ST):
        row = tl.load(idx + s + n).to(tl.int64) * 128 + hh * W
        acc += tl.load(kc + row[:, None] + c[None, :]).to(tl.float32) + tl.load(vc + row[:, None] + c[None, :]).to(tl.float32)
    tl.store(out + (pid * H + hh) * W + c, tl.sum(acc, 0))
N = 2752032; R = 32 * 2048
kc = torch.randint(0, 255, (N, 128), dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
vc = torch.randint(0, 255, (N, 128), dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
idx = torch.randperm(N, device="cuda")[:R].int()
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
res = []
lk = open('/tmp/okc-gpu.lock', 'w'); fcntl.flock(lk, fcntl.LOCK_EX)
for W, H in ((128, 1), (64, 2)):
    for chunk in (128, 256, 512):
        for bn, w, st in [(32, 1, 3), (32, 2, 5), (32, 1, 5), (64, 2, 3), (64, 4, 4), (16, 1, 6)]:
            if bn > chunk: continue
            out = torch.empty((R // chunk * H, W), device="cuda")
            f = lambda: g[(R // chunk, H)](kc, vc, idx, out, CHUNK=chunk, BN=bn, ST=st, W=W, H=H, num_warps=w)
            f(); torch.cuda.synchronize(); ts = []
            for _ in range(15):
                flush.zero_(); a = torch.cuda.Event(True); b = torch.cuda.Event(True)
                a.record(); f(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1000)
            res.append((round(statistics.median(ts), 2), W, chunk, bn, w, st))
    best = sorted(r for r in res if r[1] == W)[:4]
    print(W, best)

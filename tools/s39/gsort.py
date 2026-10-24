"""Does reading each program's random rows in sorted physical order raise gather bandwidth?
Pool = 32 requests x 80000 rows, randperm split into requests (as reference.py); program p takes a
contiguous logical chunk. Modes: random order / each chunk sorted. Times under the GPU lock."""
import torch, triton, triton.language as tl, statistics, fcntl, sys
@triton.jit
def g(kc, vc, idx, out, CHUNK: tl.constexpr, BN: tl.constexpr, ST: tl.constexpr):
    pid = tl.program_id(0)
    c = tl.arange(0, 128); n = tl.arange(0, BN)
    acc = tl.zeros([BN, 128], tl.float32)
    for s in tl.range(pid * CHUNK, pid * CHUNK + CHUNK, BN, num_stages=ST):
        row = tl.load(idx + s + n).to(tl.int64) * 128
        acc += tl.load(kc + row[:, None] + c[None, :]).to(tl.float32) + tl.load(vc + row[:, None] + c[None, :]).to(tl.float32)
    tl.store(out + pid * 128 + c, tl.sum(acc, 0))
B, L = 32, 80000
N = B * L
kc = torch.randint(0, 255, (N, 128), dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
vc = torch.randint(0, 255, (N, 128), dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
perm = torch.randperm(N, device="cuda").int()
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
lock = open("/tmp/okc-gpu.lock", "w")
for chunk in [int(x) for x in (sys.argv[1] if len(sys.argv) > 1 else '640,2560,5120').split(',')]:
    srt = perm.view(-1, chunk).sort(1).values.flatten().contiguous()
    ar = torch.arange(N, device="cuda").int()
    modes = {"random": perm, "sorted": srt, "arange": ar,
             # lockstep sweep: program p reads rows p, p + P, p + 2P ... (all programs at the same region together)
             "strided": ar.view(chunk, -1).t().contiguous().flatten()}
    for name in (sys.argv[2] if len(sys.argv) > 2 else "random,sorted").split(","):
        idx = modes[name]
        best = []
        for bn, w, st in [(32, 1, 3), (32, 2, 3), (64, 2, 3), (64, 4, 4), (32, 1, 4)]:
            out = torch.empty((N // chunk, 128), device="cuda")
            f = lambda: g[(N // chunk,)](kc, vc, idx, out, CHUNK=chunk, BN=bn, ST=st, num_warps=w)
            f(); torch.cuda.synchronize(); ts = []
            fcntl.flock(lock, fcntl.LOCK_EX)
            for _ in range(15):
                flush.zero_(); a = torch.cuda.Event(True); b = torch.cuda.Event(True)
                a.record(); f(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1000)
            fcntl.flock(lock, fcntl.LOCK_UN)
            best.append((round(statistics.median(ts), 1), bn, w, st))
        best.sort()
        t = best[0][0]
        print(f"chunk {chunk} {name}: {t:.1f} us = {2 * N * 128 / t / 1e6:.2f} TB/s {best[:3]}", flush=True)

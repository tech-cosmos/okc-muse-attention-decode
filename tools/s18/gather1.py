"""Batch-1 full-layer gather floor: 80000 random 128-byte K and V rows, graph replay, cold L2, minus empty replay."""
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

import fcntl, statistics
N = 80000
kc = torch.randint(0, 255, (N, 128), dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
vc = torch.randint(0, 255, (N, 128), dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
idx = torch.randperm(N, device="cuda").int()
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
def tg(f, reps=30):
    f(); torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        f()
    ts = []
    with open("/tmp/okc-gpu.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        for _ in range(reps):
            flush.zero_(); a = torch.cuda.Event(True); b = torch.cuda.Event(True)
            a.record(); gr.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1000)
    return statistics.median(ts)
e = torch.empty(1, device="cuda")
empty = tg(lambda: e.zero_())
res = []
for chunk in (160, 320, 640, 1280):
    for bn, w, st in [(32, 1, 3), (32, 1, 6), (32, 2, 4), (64, 2, 3), (64, 4, 4), (64, 4, 6), (128, 4, 3), (128, 4, 6), (16, 1, 6)]:
        if chunk % bn: continue
        out = torch.empty((N // chunk, 128), device="cuda")
        f = lambda: g[(N // chunk,)](kc, vc, idx, out, CHUNK=chunk, BN=bn, ST=st, num_warps=w)
        res.append((tg(f) - empty, chunk, bn, w, st))
res.sort()
for r in res[:8]: print("%.2f us chunk %d bn %d warps %d stages %d" % r)

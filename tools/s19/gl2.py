"""tools/gather.py's kernel at decode 32 full size: random rows vs rows folded into 4096 (L2-resident), one config per argv."""
import os, sys, torch, triton, triton.language as tl
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import emu  # noqa: E402
@triton.jit
def g(kc, vc, idx, out, CHUNK: tl.constexpr, BN: tl.constexpr, ST: tl.constexpr, CVT: tl.constexpr):
    pid = tl.program_id(0)
    c = tl.arange(0, 128); n = tl.arange(0, BN)
    acc = tl.zeros([BN, 128], tl.float32)
    for s in tl.range(pid * CHUNK, pid * CHUNK + CHUNK, BN, num_stages=ST):
        row = tl.load(idx + s + n).to(tl.int64) * 128
        if CVT:
            acc += tl.load(kc + row[:, None] + c[None, :]).to(tl.float32) + tl.load(vc + row[:, None] + c[None, :]).to(tl.float32)
        else:
            acc += (tl.load(kc + row[:, None] + c[None, :]).to(tl.int8, bitcast=True) ^ tl.load(vc + row[:, None] + c[None, :]).to(tl.int8, bitcast=True)).to(tl.float32)
    tl.store(out + pid * 128 + c, tl.sum(acc, 0))
N = 32 * 80000
kc = torch.randint(0, 255, (N, 128), dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
vc = torch.randint(0, 255, (N, 128), dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
idx = torch.randperm(N, device="cuda").int()
x = torch.zeros(1, device="cuda")
for blk in (0, 40):
    empty = emu.time_blocked(lambda: x.add_(1), blk, reps=60)
    for chunk, bn, w, st in [(2560, 64, 2, 3), (2560, 16, 1, 6), (2560, 32, 1, 4)]:
        out = torch.empty((N // chunk, 128), device="cuda")
        for name, ix in (("random", idx), ("L2", idx % 4096)):
            f = lambda: g[(N // chunk,)](kc, vc, ix, out, CHUNK=chunk, BN=bn, ST=st, CVT=True, num_warps=w)
            t = emu.time_blocked(f, blk, 20) - empty
            print(f"sms {132-blk} chunk {chunk} bn {bn} w {w} st {st} {name:6s} {t:7.1f} us", flush=True)

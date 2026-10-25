"""Batch-1 full gather floor with bench.py's judge-like timing (read-flush, minus empty graph).
Gathers the real decode-1-full rows (page table of seed 99) K+V, 128 B each, through the page table."""
import sys; sys.path.insert(0, "tools"); sys.path.insert(0, ".")
import torch, triton, triton.language as tl
import bench

@triton.jit
def g(kc, vc, pt, out, n, CHUNK: tl.constexpr, BN: tl.constexpr, ST: tl.constexpr, DENSE: tl.constexpr):
    pid = tl.program_id(0)
    c = tl.arange(0, 128); r = tl.arange(0, BN)
    acc = tl.zeros([BN, 128], tl.float32)
    for s in tl.range(pid * CHUNK, pid * CHUNK + CHUNK, BN, num_stages=ST):
        m = s + r < n
        if DENSE:
            row = (s + r).to(tl.int64) * 128
        else:
            row = tl.load(pt + s + r, mask=m, other=0).to(tl.int64) * 128
        acc += tl.load(kc + row[:, None] + c[None, :], mask=m[:, None], other=0.).to(tl.float32) \
            + tl.load(vc + row[:, None] + c[None, :], mask=m[:, None], other=0.).to(tl.float32)
    tl.store(out + pid * 128 + c, tl.sum(acc, 0))

c = dict(batch=1, ctx=80000, heads="tp8", layer="full", page=1, q_len=1, dist="coherent")
d = bench.data_for("decode 1", c)
kc = d["k_cache"].view(torch.uint8).view(-1, 128).view(torch.float8_e4m3fn)
vc = d["v_cache"].view(torch.uint8).view(-1, 128).view(torch.float8_e4m3fn)
pt = d["page_table"][0].contiguous()
n = int(d["seq_lens"][0])
print("rows", n, "cache rows", kc.shape[0])
emp = bench.empty_cost()
res = []
for dense in (False, True):
    for chunk in (64, 128, 160, 256, 320, 640):
        for bn, w, st in [(32, 1, 2), (32, 1, 4), (16, 1, 4), (32, 2, 3), (64, 2, 3), (64, 4, 4), (128, 4, 3)]:
            if chunk % bn: continue
            P = triton.cdiv(n, chunk)
            out = torch.empty((P, 128), device="cuda")
            f = lambda: g[(P,)](kc, vc, pt, out, n, CHUNK=chunk, BN=bn, ST=st, DENSE=dense, num_warps=w)
            res.append((bench.time_graph(f) - emp, dense, chunk, bn, w, st))
    rs = sorted(r for r in res if r[1] == dense)
    for r in rs[:5]: print("%.2f us dense %d chunk %d bn %d warps %d stages %d" % r)

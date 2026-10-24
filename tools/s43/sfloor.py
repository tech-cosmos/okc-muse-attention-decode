"""Graph-timed (bench.time_graph, cold L2, minus empty) gather floor for batched sliding shapes on the real
inputs: each program reads K and V rows of its chunk of the request's last 2047 keys through the page table.
sfloor.py "<shape>" -> table of (us, W, splits, bn, warps, stages)"""
import sys, torch, triton, triton.language as tl
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench

@triton.jit
def g(kc, vc, pt, seq, out, MAXP: tl.constexpr, S: tl.constexpr, WIN: tl.constexpr, BN: tl.constexpr,
      ST: tl.constexpr, W: tl.constexpr, H: tl.constexpr):
    sp = tl.program_id(0); b = tl.program_id(1); hh = tl.program_id(2)
    n = tl.arange(0, BN); c = tl.arange(0, W)
    sl = tl.load(seq + b)
    lo0 = tl.maximum(sl - WIN, 0)
    chunk = (sl - lo0 + S - 1) // S
    lo = lo0 + sp * chunk
    hi = tl.minimum(lo + chunk, sl)
    acc = tl.zeros([BN, W], tl.float32)
    for s in tl.range(lo, hi, BN, num_stages=ST):
        t = s + n
        pg = tl.load(pt + b.to(tl.int64) * MAXP + t, mask=t < hi, other=0).to(tl.int64)
        row = pg * 128 + hh * W
        acc += tl.load(kc + row[:, None] + c[None, :]).to(tl.float32) + tl.load(vc + row[:, None] + c[None, :]).to(tl.float32)
    tl.store(out + ((b * S + sp) * H + hh) * W + c, tl.sum(acc, 0))

shape = sys.argv[1]
for ld, cc in bench.SCORED:
    if f"{ld} {cc['layer']}" == shape:
        d = bench.data_for(ld, cc)
B = d["page_table"].shape[0]
kc = d["k_cache"].view(-1, 128); vc = d["v_cache"].view(-1, 128)
pt = d["page_table"]; seq = d["seq_lens"]
print("page_table", pt.shape, pt.dtype, "k_cache", d["k_cache"].shape)
e = bench.empty_cost()
res = []
for W, H in ((128, 1), (64, 2)):
    for S in (4, 8, 16, 32):
        for bn, w, st in [(32, 1, 3), (32, 2, 5), (32, 1, 5), (64, 2, 3), (64, 4, 4), (16, 1, 6), (32, 4, 4), (64, 4, 2)]:
            out = torch.empty((B * S * H, W), device="cuda")
            f = lambda: g[(S, B, H)](kc, vc, pt, seq, out, MAXP=pt.shape[1], S=S, WIN=2047, BN=bn, ST=st, W=W, H=H,
                                     num_warps=w)
            res.append((round(bench.time_graph(f) - e, 2), W, S, bn, w, st))
for W in (128, 64):
    print(W, sorted(r for r in res if r[1] == W)[:6])

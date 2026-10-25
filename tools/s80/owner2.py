import sys, torch, statistics, fcntl, triton, triton.language as tl
sys.path.insert(0, "/home/daytona/repo")
import reference
@triton.jit
def _own(pt, seq, owner, MP: tl.constexpr, BLK: tl.constexpr, DO: tl.constexpr):
    b = tl.program_id(1)
    t = tl.program_id(0) * BLK + tl.arange(0, BLK)
    ok = t < tl.load(seq + b)
    ph = tl.load(pt + b.to(tl.int64) * MP + t, mask=ok, other=0)
    if DO == 1:
        tl.store(owner + ph, b.to(owner.dtype.element_ty), mask=ok)
    else:
        tl.store(owner + b.to(tl.int64) * MP + t, (ph & 127).to(owner.dtype.element_ty), mask=ok)
d = reference.generate_input(batch=32, ctx=80000, heads="tp8", layer="full", seed=1, spread="agentic", dist="coherent", page=1)
pt, seq = d["page_table"], d["seq_lens"]
B, MP = pt.shape
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
lk = open("/tmp/okc-gpu.lock", "w")
def T(f, warm=False):
    g = torch.cuda.CUDAGraph(); f(); torch.cuda.synchronize()
    with torch.cuda.graph(g): f()
    ts = []
    fcntl.flock(lk, fcntl.LOCK_EX)
    for _ in range(15):
        if not warm: flush.zero_()
        a = torch.cuda.Event(True); e = torch.cuda.Event(True)
        a.record(); g.replay(); e.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(e) * 1000)
    fcntl.flock(lk, fcntl.LOCK_UN)
    return statistics.median(ts)
for dt in (torch.int8, torch.int32):
    owner = torch.empty(max(d["k_cache"].shape[0], B * MP), dtype=dt, device="cuda")
    print(dt, "memset", f"{T(lambda: owner.fill_(-1)):.1f}",
          "scatter cold", f"{T(lambda: _own[(triton.cdiv(MP, 1024), B)](pt, seq, owner, MP=MP, BLK=1024, DO=1, num_warps=4)):.1f}",
          "scatter warm", f"{T(lambda: _own[(triton.cdiv(MP, 1024), B)](pt, seq, owner, MP=MP, BLK=1024, DO=1, num_warps=4), True):.1f}",
          "seq-store cold", f"{T(lambda: _own[(triton.cdiv(MP, 1024), B)](pt, seq, owner, MP=MP, BLK=1024, DO=0, num_warps=4)):.1f}")

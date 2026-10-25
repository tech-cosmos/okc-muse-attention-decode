"""Price owner[phys] = request id (int8) per call: memset + scatter from the page table (H100, decode 32 full)."""
import sys, torch, statistics, fcntl, triton, triton.language as tl
sys.path.insert(0, "/home/daytona/repo")
import reference
@triton.jit
def _own(pt, seq, owner, MP: tl.constexpr, BLK: tl.constexpr):
    b = tl.program_id(1)
    t = tl.program_id(0) * BLK + tl.arange(0, BLK)
    ok = t < tl.load(seq + b)
    ph = tl.load(pt + b.to(tl.int64) * MP + t, mask=ok, other=0)
    tl.store(owner + ph, b.to(tl.int8), mask=ok)
if __name__ != "__main__": raise SystemExit
d = reference.generate_input(batch=32, ctx=80000, heads="tp8", layer="full", seed=1, spread="agentic", dist="coherent", page=1)
pt, seq = d["page_table"], d["seq_lens"]
B, MP = pt.shape
owner = torch.empty(d["k_cache"].shape[0], dtype=torch.int8, device="cuda")
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
lk = open("/tmp/okc-gpu.lock", "w")
for blk, w in [(1024, 4), (2048, 4), (4096, 8)]:
    def f():
        owner.fill_(-1)
        _own[(triton.cdiv(MP, blk), B)](pt, seq, owner, MP=MP, BLK=blk, num_warps=w)
    f(); torch.cuda.synchronize()
    ref = torch.full_like(owner, -1); valid = torch.arange(MP, device="cuda")[None] < seq[:, None]
    ref[pt[valid].long()] = torch.arange(B, device="cuda")[:, None].expand(B, MP)[valid].to(torch.int8)
    assert torch.equal(ref, owner)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): f()
    ts = []
    fcntl.flock(lk, fcntl.LOCK_EX)
    for _ in range(15):
        flush.zero_(); a = torch.cuda.Event(True); e = torch.cuda.Event(True)
        a.record(); g.replay(); e.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(e) * 1000)
    fcntl.flock(lk, fcntl.LOCK_UN)
    print(f"memset+scatter BLK {blk} w {w}: {statistics.median(ts):.1f} us (graph, flushed; incl. ~2 launches)")

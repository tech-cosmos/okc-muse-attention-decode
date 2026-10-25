import sys, torch, statistics, fcntl
sys.path.insert(0, "/home/daytona/repo"); sys.path.insert(0, "/home/daytona/repo/tools/s80")
import reference, bwk
d = reference.generate_input(batch=32, ctx=80000, heads="tp8", layer="full", seed=1, spread="agentic", dist="coherent", page=1)
pt, seq = d["page_table"], d["seq_lens"]
B, MP = pt.shape
kc8, vc8 = d["k_cache"].view(-1), d["v_cache"].view(-1)
valid = torch.arange(MP, device="cuda")[None] < seq[:, None]
key = torch.where(valid, torch.arange(B, device="cuda")[:, None].long() * (1 << 32) + pt.long(), 1 << 62).flatten()
rs = (torch.sort(key).values[: int(seq.sum())] & 0xFFFFFFFF).int().contiguous()
pos = torch.where(valid, torch.arange(B * MP, device="cuda").view(B, MP), B * MP)
rq = pt.flatten()[torch.sort(pos.flatten()).values[: int(seq.sum())]].contiguous()  # request order, unsorted
key1 = torch.where(valid, pt.long(), 1 << 62).flatten()
gs = torch.sort(key1).values[: int(seq.sum())].int().contiguous()
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
nsm = torch.cuda.get_device_properties(0).multi_processor_count
lk = open("/tmp/okc-gpu.lock", "w")
def t(f):
    f(); torch.cuda.synchronize(); ts = []
    fcntl.flock(lk, fcntl.LOCK_EX)
    for _ in range(11):
        flush.zero_(); a = torch.cuda.Event(True); b = torch.cuda.Event(True)
        a.record(); f(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1000)
    fcntl.flock(lk, fcntl.LOCK_UN)
    return statistics.median(ts)
for rep in range(2):
    P = nsm * 2; out = torch.empty((P, 128), device="cuda")
    print("gsort", t(lambda: bwk._bw_read[(P,)](kc8, vc8, gs, seq, out, B=32, B_PAD=32, IDX=True, LOCK=True, BN=64, ST=3, num_warps=4)))
    print("dense", t(lambda: bwk._bw_read[(P,)](kc8, vc8, gs, seq, out, B=32, B_PAD=32, IDX=False, LOCK=True, BN=64, ST=3, num_warps=4)))
    for K, bn, w, st in [(8, 64, 4, 3), (16, 32, 2, 3), (32, 32, 1, 4), (8, 32, 2, 3)]:
        P = K * 32; out = torch.empty((P, 128), device="cuda")
        print(f"req-sweep K={K} {bn,w,st}", t(lambda: bwk._bw_req[(P,)](kc8, vc8, rs, seq, out, B=32, B_PAD=32, BN=bn, ST=st, K=K, num_warps=w)),
              "unsorted", t(lambda: bwk._bw_req[(P,)](kc8, vc8, rq, seq, out, B=32, B_PAD=32, BN=bn, ST=st, K=K, num_warps=w)), flush=True)

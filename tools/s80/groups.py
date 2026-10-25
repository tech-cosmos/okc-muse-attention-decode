"""How many requests must share a physically-ordered sweep? Lists sorted within request groups (NG groups of B/NG
requests), concatenated by group, read in lockstep tiles. H100, decode 32 full bytes."""
import sys, torch, statistics, fcntl
sys.path.insert(0, "/home/daytona/repo"); sys.path.insert(0, "/home/daytona/repo/tools/s80")
import reference, bwk
d = reference.generate_input(batch=32, ctx=80000, heads="tp8", layer="full", seed=1, spread="agentic", dist="coherent", page=1)
pt, seq = d["page_table"], d["seq_lens"]
B, MP = pt.shape
kc8, vc8 = d["k_cache"].view(-1), d["v_cache"].view(-1)
valid = torch.arange(MP, device="cuda")[None] < seq[:, None]
big = 2**31 - 1
lists = {}
for ng in (1, 2, 4, 8, 16, 32):
    grp = (torch.arange(B, device="cuda") // (B // ng)).long()
    key = torch.where(valid, grp[:, None] * (1 << 32) + pt.long(), 1 << 62).flatten()
    lists[ng] = (torch.sort(key).values[: int(seq.sum())] & 0xFFFFFFFF).int().contiguous()
pos = torch.where(valid, torch.arange(B * MP, device="cuda").view(B, MP), B * MP)
lists["rand"] = pt.flatten()[torch.sort(pos.flatten()).values[: int(seq.sum())]].contiguous()
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
nsm = torch.cuda.get_device_properties(0).multi_processor_count
lk = open("/tmp/okc-gpu.lock", "w")
for name, idx in lists.items():
    for bn, w, st, ps in [(64, 4, 3, 2), (32, 2, 3, 4)]:
        P = nsm * ps; out = torch.empty((P, 128), device="cuda")
        f = lambda: bwk._bw_read[(P,)](kc8, vc8, idx, seq, out, B=32, B_PAD=32, IDX=True, LOCK=True, BN=bn, ST=st, num_warps=w)
        f(); torch.cuda.synchronize(); ts = []
        fcntl.flock(lk, fcntl.LOCK_EX)
        for _ in range(11):
            flush.zero_(); a = torch.cuda.Event(True); b = torch.cuda.Event(True)
            a.record(); f(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1000)
        fcntl.flock(lk, fcntl.LOCK_UN)
        print(f"groups {name}: {statistics.median(ts):.1f} us  cfg {(bn, w, st, ps)}", flush=True)

import sys, torch, triton, statistics, fcntl
sys.path.insert(0, "/home/daytona/repo"); sys.path.insert(0, "/home/daytona/repo/tools/s80")
import reference, bwk
d = reference.generate_input(batch=32, ctx=80000, heads="tp8", layer="full", seed=1, spread="agentic", dist="coherent", page=1)
pt, seq = d["page_table"], d["seq_lens"]
kc, vc = d["k_cache"].view(torch.uint8).view(-1), d["v_cache"].view(torch.uint8).view(-1)
print("pool rows", d["k_cache"].shape, "T", int(seq.sum()), "MB", int(seq.sum()) * 256 / 1e6)
rand, gs, rs = bwk.bw_lists(pt, seq)
kc8, vc8 = d["k_cache"].view(-1), d["v_cache"].view(-1)
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
nsm = torch.cuda.get_device_properties(0).multi_processor_count
lk = open("/tmp/okc-gpu.lock", "w"); fcntl.flock(lk, fcntl.LOCK_EX)
res = []
for name, idx, IDX in (("dense", rand, False), ("rand", rand, True), ("gsort", gs, True), ("rsort", rs, True)):
    for lock in (False, True):
        for bn, w, st, ps in [(32, 2, 3, 4), (64, 4, 3, 2), (64, 4, 4, 4), (32, 1, 4, 8), (128, 4, 3, 2)]:
            P = nsm * ps
            out = torch.empty((P, 128), device="cuda")
            f = lambda: bwk._bw_read[(P,)](kc8, vc8, idx, seq, out, B=32, B_PAD=32, IDX=IDX, LOCK=lock, BN=bn, ST=st, num_warps=w)
            f(); torch.cuda.synchronize(); ts = []
            for _ in range(11):
                flush.zero_(); a = torch.cuda.Event(True); b = torch.cuda.Event(True)
                a.record(); f(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1000)
            res.append((statistics.median(ts), name, lock, bn, w, st, ps))
    best = sorted(r for r in res if r[1] == name)[:3]
    print(name, [f"{r[0]:.1f} lock={r[2]} {r[3:]}" for r in best], flush=True)

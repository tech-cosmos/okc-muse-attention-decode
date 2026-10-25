"""s84: graph-timed bucketing pass (bucket.cu) on decode 32 full; correctness vs torch."""
import sys, torch
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench, reference
from torch.utils.cpp_extension import load_inline
m = load_inline(name="s84_bucket4", cpp_sources="void bucket_run(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t);",
                cuda_sources=open("/home/daytona/repo/tools/s84/bucket.cu").read(), functions=["bucket_run"],
                extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"])
c = dict(batch=32, ctx=80000, heads="tp8", layer="full", page=1, q_len=1, dist="coherent", spread="agentic")
data = bench.data_for("decode 32", c)
pt, seq = data["page_table"], data["seq_lens"]
rows = data["k_cache"].shape[0]
print("pool rows", rows, "entries", int(seq.sum()), "pt", tuple(pt.shape))
for nw in (132, 256):
    ws = (rows + nw - 1) // nw
    cap = ws  # unique physical rows (randperm): at most WS entries per window
    gcount = torch.zeros(256, dtype=torch.int32, device="cuda")
    lists = torch.empty(nw * cap, dtype=torch.int32, device="cuda")
    for ent in (2048, 4096):
        gcount.zero_(); m.bucket_run(pt, seq, gcount, lists, ws, cap, ent, 0); torch.cuda.synchronize()
        # check: every window's count matches torch's histogram and the entries map back to that window
        valid = torch.arange(pt.shape[1], device="cuda")[None, :] < seq[:, None].long()
        ref = torch.bincount((pt[valid].long() // ws), minlength=256)[:256]
        ok = torch.equal(ref.int(), gcount)
        w = 3; e = lists[w * cap: w * cap + int(gcount[w])]
        ok = ok and bool(((pt[(e >> 20).long(), (e & 0xfffff).long()] // ws) == w).all())
        t1 = min(bench.time_graph(lambda: (gcount.zero_(), m.bucket_run(pt, seq, gcount, lists, ws, cap, ent, 0))) - bench.empty_cost() for _ in range(3))
        t2 = min(bench.time_graph(lambda: m.bucket_run(pt, seq, gcount, lists, ws, cap, ent, 0)) - bench.empty_cost() for _ in range(3))
        ab = [min(bench.time_graph(lambda: m.bucket_run(pt, seq, gcount, lists, ws, cap, ent, a)) - bench.empty_cost() for _ in range(3)) for a in (1, 2, 3)]
        print("  ablations no-global-atomics / no-list-writes / page-table-read-only: " + " ".join(f"{x:.2f}" for x in ab))
        print(f"NW {nw} WS {ws} ENT {ent}: ok {ok}; with memset {t1:.2f} us, kernel only {t2:.2f} us", flush=True)

"""s114: graph-timed atomic-free owner pass (p1.cu) on decode 32/16 full; correctness vs torch."""
import sys, torch
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench
from torch.utils.cpp_extension import load_inline
m = load_inline(name="s114_p1", cpp_sources="""
void p1_run(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t);
void mapb_run(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
void ptread_run(torch::Tensor, torch::Tensor);""",
                cuda_sources=open("/home/daytona/repo/tools/s114/p1.cu").read(), functions=["p1_run", "mapb_run", "ptread_run"],
                extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"])
NW = 132
for load, batch in (("decode 32", 32), ("decode 16", 16)):
    c = dict(batch=batch, ctx=80000, heads="tp8", layer="full", page=1, q_len=1, dist="coherent", spread="agentic")
    data = bench.data_for(load, c)
    pt, seq = data["page_table"], data["seq_lens"]
    rows = data["k_cache"].shape[0]
    ws = (rows + NW - 1) // NW
    print(f"{load}: pool rows {rows} entries {int(seq.sum())} pt {tuple(pt.shape)} WS {ws}", flush=True)
    sink = torch.zeros(4, dtype=torch.int32, device="cuda")
    t = min(bench.time_graph(lambda: m.ptread_run(pt, sink)) - bench.empty_cost() for _ in range(3))
    print(f"  page-table read alone {t:.2f} us", flush=True)
    # reference owner map
    valid = torch.arange(pt.shape[1], device="cuda")[None, :] < seq[:, None].long()
    ref = torch.full((NW * ws,), 255, dtype=torch.uint8, device="cuda")
    bidx = torch.arange(batch, device="cuda")[:, None].expand_as(pt)
    ref[pt[valid].long()] = bidx[valid].to(torch.uint8)
    for ent in (2048, 4096, 8192, 16384):
        nsl = (pt.shape[1] + ent - 1) // ent
        out = torch.empty(batch * nsl * ent, dtype=torch.int16, device="cuda")
        off = torch.empty(batch * nsl * (NW + 1), dtype=torch.int16, device="cuda")
        omap = torch.empty(NW * ws, dtype=torch.uint8, device="cuda")
        m.p1_run(pt, seq, out, off, ws, ent)
        for nt in (512, 1024):
            m.mapb_run(out, off, omap, sink, ws, nsl, batch * nsl, ent, 1, nt)
            torch.cuda.synchronize()
            ok = torch.equal(omap, ref)
            t1 = min(bench.time_graph(lambda: m.p1_run(pt, seq, out, off, ws, ent)) - bench.empty_cost() for _ in range(3))
            t2 = min(bench.time_graph(lambda: m.mapb_run(out, off, omap, sink, ws, nsl, batch * nsl, ent, 0, nt)) - bench.empty_cost() for _ in range(3))
            t3 = min(bench.time_graph(lambda: (m.p1_run(pt, seq, out, off, ws, ent), m.mapb_run(out, off, omap, sink, ws, nsl, batch * nsl, ent, 0, nt))) - bench.empty_cost() for _ in range(3))
            print(f"  ENT {ent} NT {nt}: ok {ok}; p1 {t1:.2f} us, mapb {t2:.2f} us, both {t3:.2f} us", flush=True)

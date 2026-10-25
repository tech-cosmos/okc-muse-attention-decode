"""s114: v2 owner pass (p1v + mapbv) timing and correctness."""
import sys, torch
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench
from torch.utils.cpp_extension import load_inline
m = load_inline(name="s114_p1v", cpp_sources="""
void p1v_run(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t);
void mapbv_run(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
void ptread_run(torch::Tensor, torch::Tensor);""",
                cuda_sources=open("/home/daytona/repo/tools/s114/p1.cu").read(), functions=["p1v_run", "mapbv_run", "ptread_run"],
                extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"])
NW = 132
T = lambda f: min(bench.time_graph(f) - bench.empty_cost() for _ in range(3))
for load, batch in (("decode 32", 32), ("decode 16", 16)):
    c = dict(batch=batch, ctx=80000, heads="tp8", layer="full", page=1, q_len=1, dist="coherent", spread="agentic")
    data = bench.data_for(load, c)
    pt, seq = data["page_table"], data["seq_lens"]
    rows = data["k_cache"].shape[0]
    ws = (rows + NW - 1) // NW
    sink = torch.zeros(4, dtype=torch.int32, device="cuda")
    print(f"{load}: WS {ws}; page-table read {T(lambda: m.ptread_run(pt, sink)):.2f} us", flush=True)
    valid = torch.arange(pt.shape[1], device="cuda")[None, :] < seq[:, None].long()
    ref = torch.full((NW * ws,), 255, dtype=torch.uint8, device="cuda")
    bidx = torch.arange(batch, device="cuda")[:, None].expand_as(pt)
    ref[pt[valid].long()] = bidx[valid].to(torch.uint8)
    for ent in (1024, 2048, 4096, 8192):
        nsl = (pt.shape[1] + ent - 1) // ent
        out = torch.empty(batch * nsl * ent, dtype=torch.int16, device="cuda")
        off = torch.empty(batch * nsl * (NW + 1), dtype=torch.int16, device="cuda")
        omap = torch.empty(NW * ws, dtype=torch.uint8, device="cuda")
        m.p1v_run(pt, seq, out, off, ws, ent)
        t1 = T(lambda: m.p1v_run(pt, seq, out, off, ws, ent))
        for r in (1, 2, 4):
            if r * 32 * 4 > ent * 2: continue
            m.mapbv_run(out, off, omap, sink, ws, nsl, batch * nsl, ent, 1, r)
            torch.cuda.synchronize()
            ok = torch.equal(omap, ref)
            t2 = T(lambda: m.mapbv_run(out, off, omap, sink, ws, nsl, batch * nsl, ent, 0, r))
            t3 = T(lambda: (m.p1v_run(pt, seq, out, off, ws, ent), m.mapbv_run(out, off, omap, sink, ws, nsl, batch * nsl, ent, 0, r)))
            print(f"  ENT {ent} R {r}: ok {ok}; p1v {t1:.2f} us, mapbv {t2:.2f} us, both {t3:.2f} us", flush=True)

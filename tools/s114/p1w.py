"""s114: v3 owner pass (p1w + mapbl) timing and correctness."""
import sys, torch
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench
from torch.utils.cpp_extension import load_inline
m = load_inline(name="s114_p1w", cpp_sources="""
void p1w_run(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t);
void mapbl_run(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t);
void ptread_run(torch::Tensor, torch::Tensor);""",
                cuda_sources=open("/home/daytona/repo/tools/s114/p1.cu").read(), functions=["p1w_run", "mapbl_run", "ptread_run"],
                extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"])
NW = 132
T = lambda f: min(bench.time_graph(f) - bench.empty_cost() for _ in range(3))


def magic(ws):
    p = torch.arange(1 << 22, device="cuda", dtype=torch.int64)
    for sh in range(0, 12):
        mg = -(-(1 << (32 + sh)) // ws)
        if mg >= 1 << 32: break
        if torch.equal((p * mg) >> (32 + sh), p // ws): return mg, sh
    raise RuntimeError("no magic")


for load, batch in (("decode 32", 32), ("decode 16", 16)):
    c = dict(batch=batch, ctx=80000, heads="tp8", layer="full", page=1, q_len=1, dist="coherent", spread="agentic")
    data = bench.data_for(load, c)
    pt, seq = data["page_table"], data["seq_lens"]
    rows = data["k_cache"].shape[0]
    ws = (rows + NW - 1) // NW
    mg, sh = magic(ws)
    sink = torch.zeros(4, dtype=torch.int32, device="cuda")
    print(f"{load}: WS {ws} magic {mg} >> {sh}; page-table read {T(lambda: m.ptread_run(pt, sink)):.2f} us", flush=True)
    valid = torch.arange(pt.shape[1], device="cuda")[None, :] < seq[:, None].long()
    ref = torch.full((NW * ws,), 255, dtype=torch.uint8, device="cuda")
    bidx = torch.arange(batch, device="cuda")[:, None].expand_as(pt)
    ref[pt[valid].long()] = bidx[valid].to(torch.uint8)
    for ent in (1024, 2048, 4096, 8192):
        nsl = (pt.shape[1] + ent - 1) // ent
        out = torch.empty(batch * nsl * ent, dtype=torch.int16, device="cuda")
        off = torch.empty(batch * nsl * (NW + 1), dtype=torch.int16, device="cuda")
        omap = torch.empty(NW * ws, dtype=torch.uint8, device="cuda")
        m.p1w_run(pt, seq, out, off, ws, mg, sh, ent)
        t1 = T(lambda: m.p1w_run(pt, seq, out, off, ws, mg, sh, ent))
        for nt in (256, 512, 1024):
            m.mapbl_run(out, off, omap, sink, ws, nsl, batch * nsl, ent, 1, nt)
            torch.cuda.synchronize()
            ok = torch.equal(omap, ref)
            t2 = T(lambda: m.mapbl_run(out, off, omap, sink, ws, nsl, batch * nsl, ent, 0, nt))
            t3 = T(lambda: (m.p1w_run(pt, seq, out, off, ws, mg, sh, ent), m.mapbl_run(out, off, omap, sink, ws, nsl, batch * nsl, ent, 0, nt)))
            print(f"  ENT {ent} NT {nt}: ok {ok}; p1w {t1:.2f} us, mapbl {t2:.2f} us, both {t3:.2f} us", flush=True)

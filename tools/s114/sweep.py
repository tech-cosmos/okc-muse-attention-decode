"""s114: windowed L2 sweep ceiling (gathers only) vs random order and dense, decode 32 full bytes (H100)."""
import sys, torch
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench
from torch.utils.cpp_extension import load_inline
m = load_inline(name="s114_sweep", cpp_sources="""
void sweep_run(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t, torch::Tensor);
void dense_run(torch::Tensor, torch::Tensor, int64_t, torch::Tensor);""",
                cuda_sources=open("/home/daytona/repo/tools/s114/sweep.cu").read(), functions=["sweep_run", "dense_run"],
                extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"])
T = lambda f: min(bench.time_graph(f) - bench.empty_cost() for _ in range(3))
load = sys.argv[1] if len(sys.argv) > 1 else "decode 32"
batch = int(load.split()[1])
c = dict(batch=batch, ctx=80000, heads="tp8", layer="full", page=1, q_len=1, dist="coherent", spread="agentic")
data = bench.data_for(load, c)
pt, seq = data["page_table"], data["seq_lens"]
kc, vc = data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8)
nrows_pool = kc.shape[0]
valid = torch.arange(pt.shape[1], device="cuda")[None, :] < seq[:, None].long()
allrows = pt[valid].int().contiguous()  # request order, page-table order
N = allrows.numel()
sink = torch.zeros(4, dtype=torch.int32, device="cuda")
print(f"{load}: pool rows {nrows_pool}, entries {N} ({N * 256 / 1e6:.0f} MB K+V)", flush=True)
print(f"  dense same bytes: {T(lambda: m.dense_run(kc, vc, N, sink)):.1f} us", flush=True)
for nt, P in ((512, 132 * 16), (256, 132 * 16), (512, 132 * 32)):
    bounds = torch.linspace(0, N, P + 1, device="cuda").long()
    prog = torch.searchsorted(bounds[1:], torch.arange(N, device="cuda"), right=True)
    off1 = torch.stack([bounds[:-1], bounds[1:]], 1).int().contiguous()
    t = T(lambda: m.sweep_run(kc, vc, allrows, off1, 1, nrows_pool * 128, -1, P, nt, sink))
    print(f" NT {nt} P {P}: random (page-table order) {t:.1f} us", flush=True)
    for nwin in (16, 32, 64, 128):
        wr = (nrows_pool + nwin - 1) // nwin
        win = allrows.long() // wr
        key = prog * nwin + win
        order = torch.argsort(key, stable=True)
        rows_s = allrows[order].contiguous()
        cnt = torch.bincount(key, minlength=P * nwin).view(P, nwin)
        off = torch.zeros(P, nwin + 1, dtype=torch.long, device="cuda")
        off[:, 1:] = torch.cumsum(cnt, 1)
        off = (off + bounds[:-1, None]).int().contiguous()
        res = []
        for d in (-1, 1, 2, 3):
            res.append(T(lambda: m.sweep_run(kc, vc, rows_s, off, nwin, wr * 128, d, P, nt, sink)))
        print(f"   NWIN {nwin} ({wr * 256 / 1e6:.1f} MB/window): no prefetch {res[0]:.1f}, prefetch D 1/2/3 "
              + " / ".join(f"{x:.1f}" for x in res[1:]), flush=True)

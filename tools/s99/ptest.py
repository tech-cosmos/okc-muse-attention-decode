"""psort prototype: correctness (same multiset per program range, outputs) + timing of decode 16/32 full with
_gd_full reading the psorted table, the pre-pass inside the timed call."""
import sys, os, math; sys.path.insert(0, "tools"); sys.path.insert(0, "."); sys.path.insert(0, "tools/s99")
import torch, submission, bench
from torch.utils.cpp_extension import load_inline
from psort_src import SRC
M = load_inline(name="s99_psort", cpp_sources="void psort(torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t);",
                cuda_sources=SRC, functions=["psort"], extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"])
def ranges(seq, plan):
    B = len(seq); BN = plan["block_n"]
    if plan.get("dbal"):
        NP = plan["dprogs"]; tl = [(s + BN - 1) // BN for s in seq]; T = max(sum(tl), 1)
        excl = [sum(tl[:b]) for b in range(B)]
        c = [b + (NP - B) * excl[b] // T for b in range(B)] + [NP]
        nsp = [c[b + 1] - c[b] for b in range(B)]
    else:
        nsp = [plan["splits"]] * B
    return [[(lo, min(lo + math.ceil(math.ceil(seq[b] / nsp[b]) / BN) * BN, seq[b]))
             for lo in range(0, seq[b], math.ceil(math.ceil(seq[b] / nsp[b]) / BN) * BN)] for b in range(B)]
SHB = int(os.environ.get("SHB", "6"))  # log2 buckets
def run(data, plan, pt2):
    pt = data["page_table"]
    sh = max(0, (data["k_cache"].shape[0] - 1).bit_length() - SHB)
    dbal = bool(plan.get("dbal"))
    M.psort(pt, data["seq_lens"], pt2, plan["splits"], plan["block_n"], plan.get("dprogs", 0), int(dbal), sh)
    return submission.kernel_fn({**data, "page_table": pt2})
for idx in [int(i) for i in os.environ.get("IDX", "5,3").split(",")]:
    load, c = bench.SCORED[idx]
    data = bench.data_for(load, c)
    plan = submission._plan(data["config"], data["page_table"].shape[1], data["qkv"].device)
    pt = data["page_table"]; pt2 = torch.zeros_like(pt)
    o = run(data, plan, pt2).float(); ref = submission.kernel_fn(data).float()
    seq = data["seq_lens"].tolist(); bad = 0
    for b, rs in enumerate(ranges(seq, plan)):
        for lo, hi in rs:
            if not torch.equal(pt[b, lo:hi].sort().values, pt2[b, lo:hi].sort().values): bad += 1
    sh = max(0, (data["k_cache"].shape[0] - 1).bit_length() - SHB)
    mono = sum(int(((pt2[b, lo:hi] >> sh).diff() < 0).sum()) for b, rs in enumerate(ranges(seq, plan)) for lo, hi in rs)
    print(f"{load}: bad ranges {bad}, bucket descents {mono}, max|out-ref| {(o - ref).abs().max().item():.3g}", flush=True)
    for _ in range(2):
        t0 = bench.time_graph(lambda: submission.kernel_fn(data)) - bench.empty_cost()
        t1 = bench.time_graph(lambda: run(data, plan, pt2)) - bench.empty_cost()
        t2 = bench.time_graph(lambda: M.psort(pt, data["seq_lens"], pt2, plan["splits"], plan["block_n"], plan.get("dprogs", 0), int(bool(plan.get("dbal"))), sh)) - bench.empty_cost()
        print(f"   main {t0:.1f}  psort+main {t1:.1f}  psort alone {t2:.2f}", flush=True)

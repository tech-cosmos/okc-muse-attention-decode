"""Ceiling probe v2 (diag only): sort page-table entries by physical row inside EXACTLY each _gd_full program's
key range (decode 16: 63 equal splits; decode 32: dbal ranges), sorting outside the timed call. MODE=rev sorts
descending in odd programs (control), MODE=shuf shuffles inside each range (control for the sort itself)."""
import sys, os, math; sys.path.insert(0, "tools"); sys.path.insert(0, ".")
import torch, submission, bench
def ranges(seq, plan):
    B = len(seq); BN = plan["block_n"]
    if plan.get("dbal"):
        NP = plan["dprogs"]; tl = [(s + BN - 1) // BN for s in seq]; T = max(sum(tl), 1)
        excl = [sum(tl[:b]) for b in range(B)]
        c = [b + (NP - B) * excl[b] // T for b in range(B)] + [NP]
        nsp = [c[b + 1] - c[b] for b in range(B)]
    else:
        nsp = [plan["splits"]] * B
    out = []
    for b in range(B):
        ch = math.ceil(math.ceil(seq[b] / nsp[b]) / BN) * BN
        out.append([(lo, min(lo + ch, seq[b])) for lo in range(0, seq[b], ch)])
    return out
mode = os.environ.get("MODE", "sort")
for idx in [int(i) for i in os.environ.get("IDX", "5,3").split(",")]:
    load, c = bench.SCORED[idx]
    data = bench.data_for(load, c)
    plan = submission._plan(data["config"], data["page_table"].shape[1], data["qkv"].device)
    seq = data["seq_lens"].tolist()
    orig = data["page_table"]
    pt = orig.clone()
    for b, rs in enumerate(ranges(seq, plan)):
        for lo, hi in rs:
            r = orig[b, lo:hi]
            if mode == "shuf":
                r = r[torch.randperm(hi - lo, device=r.device)]
            elif mode.startswith("sh"):
                sh = int(mode[2:]); r = r[torch.sort(r >> sh, stable=True).indices]
            else:
                r = r.sort().values
            pt[b, lo:hi] = r
    ref = submission.kernel_fn(data).float()
    for name, p in (("orig", orig), (mode, pt), ("orig", orig), (mode, pt)):
        data["page_table"] = p
        o = submission.kernel_fn(data).float()
        ts = [bench.time_graph(lambda: submission.kernel_fn(data)) - bench.empty_cost() for _ in range(3)]
        print(f"{load} {name:5s} err={(o - ref).abs().max().item():.3g} us " + " ".join(f"{t:.1f}" for t in ts), flush=True)
    data["page_table"] = orig

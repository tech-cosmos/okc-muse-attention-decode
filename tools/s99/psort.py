"""Ceiling probe (diag only): decode 16/32 full with each request's page-table row sorted by physical row
inside blocks of X logical positions (sorting outside the timed call). Within a program's chunk the order of
keys doesn't change the math, so this measures what an in-kernel per-chunk sort could buy (lockstep sweep)."""
import sys, os; sys.path.insert(0, "tools"); sys.path.insert(0, ".")
import torch, submission, reference, bench
def sorted_pt(data, X):
    pt = data["page_table"].clone()
    seq = data["seq_lens"] if "seq_lens" in data else None
    B, W = pt.shape
    out = pt.clone()
    for b in range(B):
        n = int(data["seq_lens"][b]) if seq is not None else W
        row = pt[b, :n]
        nb = (n + X - 1) // X
        pad = nb * X - n
        r = torch.cat([row, torch.full((pad,), 2**31 - 1, dtype=row.dtype, device=row.device)]).view(nb, X)
        r = r.sort(dim=1).values.view(-1)[:n] if X > 0 else row
        out[b, :n] = r
    return out
for idx in [int(i) for i in os.environ.get("IDX", "5,3").split(",")]:
    load, c = bench.SCORED[idx]
    data = bench.data_for(load, c)
    print(load, c["layer"], {k: (tuple(v.shape) if torch.is_tensor(v) else v) for k, v in data.items() if k in ("page_table", "seq_lens")}, flush=True)
    orig = data["page_table"]
    ref = submission.kernel_fn(data).float()
    for X in [int(x) for x in os.environ.get("XS", "0,128,512,2048,8192,100000").split(",")]:
        data["page_table"] = orig if X == 0 else sorted_pt({**data, "page_table": orig}, X)
        o = submission.kernel_fn(data).float()
        err = (o - ref).abs().max().item()
        ts = [bench.time_graph(lambda: submission.kernel_fn(data)) - bench.empty_cost() for _ in range(3)]
        print(f"  X={X:6d} err={err:.3g} us " + " ".join(f"{t:.1f}" for t in ts), flush=True)
    data["page_table"] = orig

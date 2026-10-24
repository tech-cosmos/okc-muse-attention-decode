"""How much does per-request length spread cost the full layers? Times each batched full shape as is (agentic
lengths), with every request cut to the shortest length, and reports the balanced estimate t_min * mean / min.
python tools/s21/imbal.py [filter=full]"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import bench, submission, torch  # noqa: E402

filt = sys.argv[1] if len(sys.argv) > 1 else "full"
for load, c in bench.SCORED:
    if filt not in f"{load} {c['layer']}" or c["batch"] == 1:
        continue
    data = dict(bench.data_for(load, c))
    if os.environ.get("L2"):
        data["page_table"] = data["page_table"] % 4096
    seq = data["seq_lens"]
    t = bench.time_graph(lambda: submission.kernel_fn(data)) - bench.empty_cost()
    lo, mean, hi = seq.min().item(), seq.float().mean().item(), seq.max().item()
    res = []
    for v in (lo, lo):
        d2 = dict(data)
        d2["seq_lens"] = torch.full_like(seq, v)
        res.append(bench.time_graph(lambda: submission.kernel_fn(d2)) - bench.empty_cost())
    print(f"{load:10s} {c['layer']:8s} agentic {t:7.2f}  all=min({lo}) {res[0]:7.2f}  again {res[1]:7.2f}  "
          f"balanced est {res[0] * mean / lo:7.2f}  (mean {mean:.0f})", flush=True)

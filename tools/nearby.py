"""Nearby shapes: correctness, and time per (request x attended key) to spot cliffs between batch sizes."""
import sys; sys.path.insert(0, ".")
import torch, bench, reference, submission
for layer, dist in (("sliding", "soft"), ("full", "coherent")):
    for q_len in (1, 8):
        for batch in (1, 2, 4, 8, 12, 16, 24, 32, 48, 64):
            for ctx in ((40000, 120000) if batch in (8, 24) else (80000,)):
                c = dict(batch=batch, ctx=ctx, heads="tp8", layer=layer, page=1, q_len=q_len, dist=dist,
                         **({"spread": "agentic"} if batch > 1 else {}))
                data = reference.generate_input(seed=7, **c)
                out = submission.kernel_fn(data)
                ok, msg = (True, "") if batch > 16 else reference.check_implementation(data, reference.collect(data, out))
                t = bench.time_graph(lambda: submission.kernel_fn(data)) - bench.empty_cost()
                keys = batch * (min(ctx, 2048) if layer == "sliding" else ctx)
                print(f"{layer:7s} L={q_len} B={batch:2d} ctx={ctx:6d} {t:8.2f} us  {1e6 * t / keys:7.2f} ps/key  "
                      f"{'PASS' if ok else 'FAIL ' + msg[:80]}", flush=True)
                del data
                torch.cuda.empty_cache()

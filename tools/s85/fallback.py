"""s85: SPEC fallback correctness: batch-1 page-1 cases with page tables wider than seq + L (spec misses)."""
import sys, torch; sys.path.insert(0, "/home/daytona/repo")
import reference, submission
for c in [dict(layer="sliding", q_len=1), dict(layer="sliding", q_len=8), dict(layer="full", q_len=8), dict(layer="full", q_len=1)]:
    for extra in (0, 1, 7, 3000):
        data = reference.generate_input(seed=7, batch=1, ctx=80000, heads="tp8", page=1, dist="soft", **c)
        if extra:
            pt = data["page_table"]; data["page_table"] = torch.cat([pt, torch.full((1, extra), -1, dtype=pt.dtype, device=pt.device)], 1)
        out = submission.kernel_fn(data); torch.cuda.synchronize()
        ok, msg = reference.check_implementation(data, reference.collect(data, out)); print(c, extra, ok, msg[:60], flush=True)

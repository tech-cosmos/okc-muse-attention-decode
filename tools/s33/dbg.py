"""Which rows/heads come out NaN on a gbal case with a plan override."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import torch, reference, submission
over = json.loads(sys.argv[1])
orig = submission._plan
submission._plan = lambda cfg, mp, dev: {**orig(cfg, mp, dev), **over}
c = dict(batch=16, ctx=20000, heads="tp8", layer="full", q_len=8, page=1, spread="agentic", dist="coherent")
data = reference.generate_input(seed=5, **c)
out = submission.kernel_fn(data).float().view(16, 8, 16, 64)
torch.cuda.synchronize()
bad = ~torch.isfinite(out).all(-1)
print("nonfinite (b, tok, head):", bad.sum().item(), "of", bad.numel())
print("per batch", bad.sum((1, 2)).tolist()); print("per head", bad.sum((0, 1)).tolist()); print("per tok", bad.sum((0, 2)).tolist())

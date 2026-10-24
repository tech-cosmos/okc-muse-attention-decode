import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import torch, reference, submission
buf = torch.zeros(4, 64, 64, device="cuda")
orig = submission._plan
submission._plan = lambda cfg, mp, dev: {**orig(cfg, mp, dev), "qreg": True, "dbg": 3, "dbg_buf": buf}
c = dict(batch=16, ctx=20000, heads="tp8", layer="full", q_len=8, page=1, spread="agentic", dist="coherent")
data = reference.generate_input(seed=5, **c)
submission.kernel_fn(data); torch.cuda.synchronize()
torch.set_printoptions(precision=2, linewidth=200, sci_mode=False)
for h in range(2):
    a, b = buf[2 * h], buf[2 * h + 1]
    print("head", h, "max diff", (a - b).abs().max().item(), "a norm", a.norm().item(), "b norm", b.norm().item())
    print(a[:4, :8]); print(b[:4, :8])
    # Find a permutation relation: does a row match some row of b?
    d = torch.cdist(a, b); print("row match", d.argmin(1)[:16].tolist(), d.min(1).values[:8].tolist())
    d = torch.cdist(a.T, b.T); print("col match", d.argmin(1)[:16].tolist(), d.min(1).values[:8].tolist())

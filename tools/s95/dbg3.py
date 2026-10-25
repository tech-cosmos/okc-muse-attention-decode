import sys; sys.path.insert(0, "tools"); sys.path.insert(0, ".")
import torch, submission, reference
orig = submission._plan
submission._plan = lambda cfg, mp, dev: {**orig(cfg, mp, dev), "red": True}
data = reference.generate_input(seed=1, batch=1, ctx=80000, heads="tp8", layer="full", page=1, q_len=1, dist="coherent")
data["k_scale"] = data["k_scale"] * 40
for i in range(3):
    d2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in data.items()}
    submission.kernel_fn(d2); torch.cuda.synchronize()
    dev = d2["qkv"].device
    buf = submission._COUNTERS[(dev, "red")]; rc = submission._COUNTERS[(dev, "r")]
    print(i, "sums nonzero", int((buf != 0).sum()), "nflag", rc[1].item(), "list[:6]", rc[2:8].tolist())
# count list entries via a run without fin
class Nop:
    def __getitem__(self, g): return lambda *a, **k: None
submission._red_fin = Nop()
d2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in data.items()}
submission.kernel_fn(d2); torch.cuda.synchronize()
n = rc[1].item(); lst = rc[2:2 + n].tolist()
print("nflag", n, "unique", len(set(lst)), "max", max(lst) if lst else None)

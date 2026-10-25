import sys; sys.path.insert(0, "tools"); sys.path.insert(0, ".")
import torch, submission, reference
orig = submission._plan
data = reference.generate_input(seed=5, batch=1, ctx=300, heads="tp8", layer="full", page=1, q_len=1, dist="coherent")
# main's partials: capture via _combine args
cap = {}
oc = submission._combine
class Cap:
    def __getitem__(self, g):
        def f(*a, **k): cap["pacc"], cap["pml"] = a[1].clone(), a[2].clone(); return oc[g](*a, **k)
        return f
submission._combine = Cap()
d2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in data.items()}
submission.kernel_fn(d2); torch.cuda.synchronize()
pml = cap["pml"].view(2, 512, 8, 2)
m, l = pml[..., 0].reshape(2, 512, 8), pml[..., 1].reshape(2, 512, 8)
ok = l > 0
print("rows m range per head", [(round(m[h // 8, :, h % 8][ok[h // 8, :, h % 8]].min().item(), 1), round(m[h // 8, :, h % 8][ok[h // 8, :, h % 8]].max().item(), 1)) for h in range(16)])
lsum = torch.where(ok, l * torch.exp2(m.clamp(-200, 200)), 0).sum(1).flatten()
# red sums without the fin kernel
of = submission._red_fin
class Nop:
    def __getitem__(self, g): return lambda *a, **k: None
submission._red_fin = Nop()
submission._plan = lambda cfg, mp, dev: {**orig(cfg, mp, dev), "red": True}
d2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in data.items()}
submission.kernel_fn(d2); torch.cuda.synchronize()
buf = submission._COUNTERS[(d2["qkv"].device, "red")]
NR = submission.RED_NR
lr = buf[NR * 16 * 64: NR * 16 * 64 + NR * 16].view(NR, 16).sum(0)
print("l ratio red/expected", (lr / lsum).tolist())
print("list", submission._COUNTERS[(d2["qkv"].device, "r")][:8].tolist())

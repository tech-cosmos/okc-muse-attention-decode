import sys, torch
sys.path.insert(0, ".")
import reference, submission as S
c = dict(batch=16, ctx=9000, heads="tp8", layer="full", q_len=8, dist="soft", spread="mixed", page=int(sys.argv[1]) if len(sys.argv) > 1 else 1)
data = reference.generate_input(seed=5, **c)
cfg = data["config"]; plan = S._plan(cfg, data["page_table"].shape[1], torch.device("cuda"))
batch, hq, hkv, d = cfg["batch"], 16, 2, 64
progs = 2 * S._num_sms(torch.device("cuda")); nid = progs + 2 * batch
qk = cfg["sm_scale"] * cfg["qk_scale_factor"] * S.LOG2E
res = []
for nat in (0, 1):
    pacc = torch.full((hkv, nid, 64, d), float("nan"), device="cuda"); pml = torch.full((hkv, nid, 64, 2), float("nan"), device=torch.device("cuda"))
    if nat:
        S._NV.nverify(data["qkv"], data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"], data["seq_lens"], pacc, pml, qk, progs, nid, cfg["page_size"])
    else:
        S._gw_verify[(progs, 1, 1)](data["qkv"], data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"], data["page_table"], data["seq_lens"],
            data["cache_loc"], pacc, pml, qk, HQ=hq, HKV=hkv, G=8, L=8, D=d, PAGE=cfg["page_size"], MAX_PAGES=data["page_table"].shape[1], S=progs,
            BN=64, NS=3, PW=4, PREGS=40, CREGS=96, LAG=1, BAL=True, B=batch, B_PAD=32, NID=nid, NP=progs, num_warps=4, maxnreg=80, launch_pdl=True)
    torch.cuda.synchronize()
    res.append((pacc, pml))
(a0, m0), (a1, m1) = res
ok = ~torch.isnan(m0[..., 0])
print("valid ids", ok[0].sum().item(), "nan in native where gluon valid", torch.isnan(m1[..., 0])[ok].sum().item())
o0 = a0 / m0[..., 1:2]; o1 = a1 / m1[..., 1:2]
for h in range(2):
    for i in [0, 1, 2, 100, 263]:
        if ok[h, i, 0]:
            print(h, i, "m", m0[h, i, :3, 0].tolist(), m1[h, i, :3, 0].tolist(), "rel", ((o0[h, i] - o1[h, i]).norm() / o0[h, i].norm()).item())
dm = (m0[..., 0] - m1[..., 0] - 8).abs()
for h in range(2):
    v = ok[h]
    print("head", h, "m mismatch frac", (dm[h][v] > 1e-3).float().mean().item(), "rows mismatched per row index", (dm[h] > 1e-3)[v.any(1)].float().mean(0)[:16].tolist())
bad = ((dm > 1e-3) & ok).any(-1)
print("ids with any mismatch h0", bad[0].nonzero().flatten()[:20].tolist())

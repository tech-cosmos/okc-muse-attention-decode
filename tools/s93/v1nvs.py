"""s93: agent2's nvs kernel at verify 1 sliding with S key splits vs main's Triton path (H100, cold, paired rounds).
python tools/s93/v1nvs.py [S ...]"""
import sys, statistics, torch
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench, submission, reference
Ss = [int(x) for x in sys.argv[1:]] or [8, 16, 32]
src = open("/home/daytona/repo/submission.py").read()
c = [c for l, c in bench.SCORED if l == "verify 1" and c["layer"] == "sliding"][0]
data = bench.data_for("verify 1", c)
ref = reference.ref_kernel(data) if hasattr(reference, "ref_kernel") else None
orig_min = submission.NVS_MIN_BATCH
def variant(S):
    def run(d):
        cfg = d["config"]
        g = submission._verify_nvs.__globals__
        # patch S via a wrapper that rebinds the constant split choice
        return _vn(d, cfg, S)
    return run
import triton
def _vn(data, cfg, S):
    batch, length, hq, hkv, d = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    qkv, device = data["qkv"], data["qkv"].device
    chunk = triton.cdiv(triton.cdiv(cfg["window"], S), 64) * 64
    rows = length * (hq // hkv)
    pacc = torch.empty((batch, hkv, S, rows, d), dtype=torch.float32, device=device)
    pml = torch.empty((batch, hkv, S, rows, 2), dtype=torch.float32, device=device)
    out = torch.empty((batch, length, hq * d), dtype=torch.bfloat16, device=device)
    s = submission
    s._NVS.nvs(qkv, s._cos_sin(cfg["rope_theta"], d, device), data["k_cache"], data["v_cache"], data["k_scale"],
             data["v_scale"], data["cache_loc"], data["page_table"], data["seq_lens"], pacc, pml,
             cfg["sm_scale"] * cfg["qk_scale_factor"] * s.LOG2E, S, chunk, cfg["window"])
    s._combine_hb[(batch * length, hq // 2)](
        qkv, pacc, pml, data["v_scale"], out,
        HQ=hq, HKV=hkv, G=hq // hkv, L=length, M=rows, D=d, S=S, S_PAD=triton.next_power_of_2(S), SB=S, HB=2,
        ONE=s.MERGE1, PDL=s.USE_PDL, num_warps=1, launch_pdl=s.USE_PDL)
    return out
base = submission.kernel_fn(data).float()
for S in Ss:
    o = _vn(data, data["config"], S).float()
    print("S", S, "max |nvs - main|", (o - base).abs().max().item(), "rel", ((o - base).norm() / base.norm()).item())
e = bench.empty_cost()
res = {S: [] for S in Ss}; res["main"] = []
for r in range(7):
    res["main"].append(bench.time_graph(lambda: submission.kernel_fn(data)) - e)
    for S in Ss:
        res[S].append(bench.time_graph(lambda: _vn(data, data["config"], S)) - e)
for k, v in res.items():
    print(k, round(statistics.median(v), 2), [round(x, 2) for x in v])

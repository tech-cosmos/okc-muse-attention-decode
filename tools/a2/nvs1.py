"""agent2 s7: nvs at S 1 (one CTA per request, no merge needed) as a proxy for a no-split wgmma decode kernel."""
import sys, re, torch
sys.path.insert(0, "/home/daytona/repo2/tools"); sys.path.insert(0, "/home/daytona/repo2")
import bench, submission
from torch.utils.cpp_extension import load_inline
src0 = submission._NVS_SRC
mods = {}
V = {"base": [], "nosm": [("softmax(s, m, l, pout, alpha, scale);\n            wg_wait<0>();", "wg_wait<0>();")],
     "nomma": [("issue_qk(s, qa, smem_u32(&sm.k[sc][0]) + h * 64);\n            fence_regs(pin);\n            issue_pv(o, pin, smem_u32(&sm.vt[sp][h][0]));", "fence_regs(pin);")],
     "notr": [("for (int q = 0; q < 4; ++q) {\n                    const int u", "for (int q = 0; q < 0; ++q) {\n                    const int u")]}
V["nocomp"] = V["nosm"] + V["nomma"]
for name, reps in V.items():
    ns, lag, pfd = 8, 3, 6
    s0 = src0
    for a, b in reps:
        assert a in s0, a
        s0 = s0.replace(a, b)
    s = s0.replace("#define NV_NS 4", f"#define NV_NS {ns}").replace("#define NV_LAG 1", f"#define NV_LAG {lag}").replace("#define NV_PFD 4", f"#define NV_PFD {pfd}")
    mods[name] = load_inline(name=f"a2nvs1_{name}_{abs(hash(s)) % 10**8}", cpp_sources="void nvs(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t, int64_t, int64_t);",
        cuda_sources=s, functions=["nvs"], extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"])
for load, c in bench.SCORED:
    if c["layer"] != "sliding" or c["q_len"] != 8 or c["batch"] == 1: continue
    data = bench.data_for(load, c); cfg = data["config"]; B = cfg["batch"]
    e = bench.empty_cost()
    res = []
    for key, m in mods.items():
        for S in (1, 4):
            ch = 2048 // S
            pacc = torch.empty((B, 2, S, 64, 64), dtype=torch.float32, device="cuda"); pml = torch.empty((B, 2, S, 64, 2), dtype=torch.float32, device="cuda")
            f = lambda: m.nvs(data["qkv"], submission._cos_sin(cfg["rope_theta"], 64, "cuda"), data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"], data["cache_loc"], data["page_table"], data["seq_lens"], pacc, pml, cfg["sm_scale"] * cfg["qk_scale_factor"] * submission.LOG2E, S, ch, 2048)
            res.append(f"{key} S{S} {bench.time_graph(f) - e:.2f}")
    print(load, " | ".join(res), flush=True)

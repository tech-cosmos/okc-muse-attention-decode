"""agent2 s8: nvs kernel-only (no combine) ablations at the shipped config, stacked variants.
python tools/a2/nvs2.py 'name|old=>new|...' ...   (name '+x' = stacked on the previous variant)"""
import sys, os, torch
sys.path.insert(0, "/home/daytona/repo2/tools"); sys.path.insert(0, "/home/daytona/repo2")
import bench, submission
from torch.utils.cpp_extension import load_inline
src0 = submission._NVS_SRC
mods = {}; prev = src0
for spec in ["base"] + sys.argv[1:]:
    parts = spec.split("|"); name = parts[0]
    s0 = prev if name.startswith("+") else src0
    for p in parts[1:]:
        a, b = p.split("=>"); a = a.replace("\\n", "\n"); b = b.replace("\\n", "\n")
        assert a in s0, (name, a)
        s0 = s0.replace(a, b)
    prev = s0
    mods[name] = load_inline(name=f"a2nvs2_{abs(hash(s0)) % 10**10}", cpp_sources="void nvs(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t, int64_t, int64_t);",
        cuda_sources=s0, functions=["nvs"], extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"])
for load, c in bench.SCORED:
    if c["layer"] != "sliding" or c["q_len"] != 8 or c["batch"] == 1: continue
    data = bench.data_for(load, c); cfg = data["config"]; B = cfg["batch"]
    e = bench.empty_cost(); cs = submission._cos_sin(cfg["rope_theta"], 64, "cuda")
    res = []
    for key, m in mods.items():
        for S in ([int(x) for x in os.environ["NVS_S"].split(",")] if "NVS_S" in os.environ else ((4,) if B == 32 else (8,))):
            ch = 2048 // S
            pacc = torch.empty((B, 2, S, 64, 64), dtype=torch.float32, device="cuda"); pml = torch.empty((B, 2, S, 64, 2), dtype=torch.float32, device="cuda")
            f = lambda: m.nvs(data["qkv"], cs, data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"], data["cache_loc"], data["page_table"], data["seq_lens"], pacc, pml, cfg["sm_scale"] * cfg["qk_scale_factor"] * submission.LOG2E, S, ch, 2048)
            ts = sorted(bench.time_graph(f) - e for _ in range(3))
            res.append(f"{key} S{S} {ts[1]:.2f}")
    print(load, " | ".join(res), flush=True)

"""agent2 s8: run one nvs variant a few times (for ncu). python tools/a2/nvs_one.py S [nogather]"""
import sys, torch
sys.path.insert(0, "/home/daytona/repo2/tools"); sys.path.insert(0, "/home/daytona/repo2")
import bench, submission
from torch.utils.cpp_extension import load_inline
s0 = submission._NVS_SRC
if "nogather" in sys.argv:
    s0 = s0.replace("                cp16(kdst + s", "                if (lim == -12345) cp16(kdst + s").replace("                cp16(vdst + s", "                if (lim == -12345) cp16(vdst + s")
m = load_inline(name=f"a2nvs1_{abs(hash(s0)) % 10**10}", cpp_sources="void nvs(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t, int64_t, int64_t);",
    cuda_sources=s0, functions=["nvs"], extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3", "-lineinfo"])
S = int(sys.argv[1])
for load, c in bench.SCORED:
    if f"{load} {c['layer']}" != "verify 32 sliding": continue
    data = bench.data_for(load, c); cfg = data["config"]; B = cfg["batch"]
    cs = submission._cos_sin(cfg["rope_theta"], 64, "cuda")
    pacc = torch.empty((B, 2, S, 64, 64), dtype=torch.float32, device="cuda"); pml = torch.empty((B, 2, S, 64, 2), dtype=torch.float32, device="cuda")
    for _ in range(4):
        m.nvs(data["qkv"], cs, data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"], data["cache_loc"], data["page_table"], data["seq_lens"], pacc, pml, cfg["sm_scale"] * cfg["qk_scale_factor"] * submission.LOG2E, S, 2048 // S, 2048)
    torch.cuda.synchronize()

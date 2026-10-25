import os, sys, torch, numpy as np
os.environ["ABL"] = "9"
sys.argv = [sys.argv[0], "none"]
sys.path.insert(0, "/home/daytona/repo/tools/s96")
import importlib.util
src = os.environ["SRC"]
from torch.utils.cpp_extension import load_inline
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench, submission
m = load_inline(name="s96_stamp_" + os.path.basename(src).split(".")[0], cpp_sources="void vtk_dbg(torch::Tensor); void vtk_verify(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, double, int64_t, int64_t);",
                cuda_sources=open(src).read(), functions=["vtk_verify", "vtk_dbg"],
                extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3", "-DABL=9"], verbose=False)
c = dict(batch=1, ctx=80000, heads="tp8", layer="sliding", page=1, q_len=8, dist="soft")
data = bench.data_for("verify 1", c); cfg = data["config"]
out = torch.empty((1, 8, 1024), dtype=torch.bfloat16, device="cuda")
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda"); sink = torch.zeros((), dtype=torch.int64, device="cuda")
res = []
for r in range(15):
    torch.sum(flush.view(torch.int64), 0, out=sink)
    m.vtk_verify(data["qkv"], submission._cos_sin(cfg["rope_theta"], 64, "cuda"), data["k_cache"], data["v_cache"], data["page_table"], data["seq_lens"], data["cache_loc"], data["k_scale"], data["v_scale"], out, 1, cfg["sm_scale"] * cfg["qk_scale_factor"] * submission.LOG2E, 1, 0)
    torch.cuda.synchronize()
    d = torch.zeros(64 * 2 * 8, dtype=torch.int64); m.vtk_dbg(d)
    a = d.view(64, 2, 8).numpy().astype(np.float64)
    nc = int(os.environ.get("NC", "8"))
    a = a[:nc]
    t0 = a[:, :, 0].min()
    res.append((a - t0) / 1000.0)
a = np.median(np.stack(res), 0)
names = ["start", "loads issued", "new tok done", "QK+mask", "PV done", "cl wait", "owner rx", "done"]
for k, nm in enumerate(names):
    for w in range(2):
        v = a[:, w, k]
        print(f"{nm:13s} {'r15w3' if w else 'r0w0 '} min {v.min():5.2f} med {np.median(v):5.2f} max {v.max():5.2f}")

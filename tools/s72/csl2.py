"""csl2.cu (cluster size NC 16 or 8): correctness + paired timing vs Triton on decode sliding. python tools/s72/csl2.py rounds"""
import os, sys
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import torch, bench, reference, submission
from torch.utils.cpp_extension import load_inline
here = os.path.dirname(os.path.abspath(__file__))
mod = load_inline("csl2_s72", cpp_sources="void csl_decode(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                  "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, double, int64_t);",
                  cuda_sources=open(os.path.join(here, "csl2.cu")).read(), functions=["csl_decode"],
                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v"], verbose=False)


def run(data, nc):
    cfg = data["config"]
    qkv = data["qkv"]
    out = torch.empty((cfg["batch"], 1, cfg["num_q_heads"] * 64), dtype=torch.bfloat16, device=qkv.device)
    mod.csl_decode(qkv, submission._cos_sin(cfg["rope_theta"], 64, qkv.device), data["k_cache"], data["v_cache"],
                   data["page_table"], data["seq_lens"], data["cache_loc"], data["k_scale"], data["v_scale"], out,
                   cfg["page_size"], cfg["sm_scale"] * cfg["qk_scale_factor"] * submission.LOG2E, nc)
    return out


cases = [dict(batch=1, ctx=10, heads="tp8", layer="sliding"),
         dict(batch=3, ctx=2047, heads="tp8", layer="sliding"),
         dict(batch=4, ctx=6000, heads="tp8", layer="sliding", dist="edges", page=1),
         dict(batch=2, ctx=6000, heads="tp8", layer="sliding", kv_scale=0.015625),
         dict(batch=32, ctx=80000, heads="tp8", layer="sliding", page=1, dist="soft", spread="agentic")]
for nc in (16, 8):
    for i, c in enumerate(cases):
        data = reference.generate_input(seed=7 + i, **c)
        out = run(data, nc)
        torch.cuda.synchronize()
        ok, msg = reference.check_implementation(data, reference.collect(data, out))
        print(nc, "PASS" if ok else "FAIL", c, msg[:60], flush=True)
rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
submission.CSL_MAX_BATCH = 0
for load, c in bench.SCORED:
    if c["layer"] != "sliding" or c["q_len"] != 1:
        continue
    d = bench.data_for(load, c)
    res = {k: [] for k in ("triton", 16, 8)}
    for _ in range(rounds):
        res["triton"].append(bench.time_graph(lambda: submission.kernel_fn(d), reps=60) - bench.empty_cost())
        for nc in (16, 8):
            res[nc].append(bench.time_graph(lambda: run(d, nc), reps=60) - bench.empty_cost())
    print(load, "sliding:", "  ".join(f"{k} {sorted(v)[len(v) // 2]:.2f}" for k, v in res.items()), flush=True)

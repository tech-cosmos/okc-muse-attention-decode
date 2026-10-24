"""Native cluster decode-sliding kernel (tools/s72/csl.cu): correctness vs reference + graph timing vs main.
python tools/s72/csl.py [reps]"""
import os, sys
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import torch, bench, reference, submission
from torch.utils.cpp_extension import load_inline
here = os.path.dirname(os.path.abspath(__file__))
mod = load_inline("csl_s72", cpp_sources="void csl_decode(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                  "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, double, int64_t);",
                  cuda_sources=open(os.path.join(here, "csl.cu")).read(), functions=["csl_decode"],
                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])


def run(data, abl=0):
    cfg = data["config"]
    qkv = data["qkv"]
    out = torch.empty((cfg["batch"], 1, cfg["num_q_heads"] * 64), dtype=torch.bfloat16, device=qkv.device)
    cs = submission._cos_sin(cfg["rope_theta"], 64, qkv.device)
    mod.csl_decode(qkv, cs, data["k_cache"], data["v_cache"], data["page_table"], data["seq_lens"], data["cache_loc"],
                   data["k_scale"], data["v_scale"], out, cfg["page_size"],
                   cfg["sm_scale"] * cfg["qk_scale_factor"] * submission.LOG2E, abl)
    return out


cases = [dict(batch=1, ctx=10, heads="tp8", layer="sliding"),
         dict(batch=1, ctx=2047, heads="tp8", layer="sliding", page=1),
         dict(batch=1, ctx=3000, heads="tp8", layer="sliding", dist="edges", page=1),
         dict(batch=1, ctx=6000, heads="tp8", layer="sliding", kv_scale=0.015625),
         dict(batch=2, ctx=5000, heads="tp8", layer="sliding", page=1, spread="mixed"),
         dict(batch=1, ctx=80000, heads="tp8", layer="sliding", page=1, dist="soft")]
for i, c in enumerate(cases):
    data = reference.generate_input(seed=7 + i, **c)
    out = run(data)
    torch.cuda.synchronize()
    ok, msg = reference.check_implementation(data, reference.collect(data, out))
    print("PASS" if ok else "FAIL", c, msg[:120], flush=True)

reps = int(sys.argv[1]) if len(sys.argv) > 1 else 3
for load, c in bench.SCORED:
    if c["layer"] != "sliding" or c["batch"] != 1 or c["q_len"] != 1:
        continue
    d = bench.data_for(load, c)
    for _ in range(reps):
        tm = bench.time_graph(lambda: submission.kernel_fn(d), reps=60) - bench.empty_cost()
        tn = bench.time_graph(lambda: run(d), reps=60) - bench.empty_cost()
        ta = [bench.time_graph(lambda: run(d, a), reps=60) - bench.empty_cost() for a in (1, 2, 3, 13)]
        print(f"{load} sliding: main {tm:.2f} us, csl {tn:.2f} us, no-merge {ta[0]:.2f}, no-gather {ta[1]:.2f}, chain-only {ta[2]:.2f}, chain-only no cluster {ta[3]:.2f}", flush=True)

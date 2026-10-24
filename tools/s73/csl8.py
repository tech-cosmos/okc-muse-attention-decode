"""Cluster-8 decode-sliding kernel (tools/s73/csl8.cu): correctness + graph timing vs main. python tools/s73/csl8.py [reps]"""
import os, sys
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import torch, bench, reference, submission
from torch.utils.cpp_extension import load_inline
here = os.path.dirname(os.path.abspath(__file__))
ABL = int(os.environ.get("ABL", "0")); ONL = int(os.environ.get("ONL", "1"))
mod = load_inline("csl8_s73a%do%d" % (ABL, ONL), cpp_sources="void csl8_decode(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                  "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, double);",
                  cuda_sources=open(os.path.join(here, "csl8.cu")).read(), functions=["csl8_decode"],
                  extra_cuda_cflags=["-O3", "-DABL=%d" % ABL, "-DONL=%d" % ONL, "-gencode=arch=compute_90a,code=sm_90a"])


def run(data):
    cfg = data["config"]
    qkv = data["qkv"]
    out = torch.empty((cfg["batch"], 1, cfg["num_q_heads"] * 64), dtype=torch.bfloat16, device=qkv.device)
    cs = submission._cos_sin(cfg["rope_theta"], 64, qkv.device)
    mod.csl8_decode(qkv, cs, data["k_cache"], data["v_cache"], data["page_table"], data["seq_lens"], data["cache_loc"],
                    data["k_scale"], data["v_scale"], out, cfg["page_size"],
                    cfg["sm_scale"] * cfg["qk_scale_factor"] * submission.LOG2E)
    return out


cases = [dict(batch=1, ctx=10, heads="tp8", layer="sliding"),
         dict(batch=1, ctx=2047, heads="tp8", layer="sliding", page=1),
         dict(batch=3, ctx=2047, heads="tp8", layer="sliding"),
         dict(batch=4, ctx=3000, heads="tp8", layer="sliding", dist="edges", page=1),
         dict(batch=1, ctx=6000, heads="tp8", layer="sliding", kv_scale=0.015625),
         dict(batch=5, ctx=5000, heads="tp8", layer="sliding", page=1, spread="mixed"),
         dict(batch=32, ctx=80000, heads="tp8", layer="sliding", page=1, spread="agentic", dist="soft")]
for i, c in enumerate(cases if ABL == 0 else []):
    data = reference.generate_input(seed=7 + i, **c)
    out = run(data)
    torch.cuda.synchronize()
    ok, msg = reference.check_implementation(data, reference.collect(data, out))
    print("PASS" if ok else "FAIL", c, msg[:120], flush=True)

reps = int(sys.argv[1]) if len(sys.argv) > 1 else 3
for load, c in bench.SCORED:
    if c["layer"] != "sliding" or c["q_len"] != 1:
        continue
    d = bench.data_for(load, c)
    for _ in range(reps):
        tm = bench.time_graph(lambda: submission.kernel_fn(d), reps=60) - bench.empty_cost()
        tn = bench.time_graph(lambda: run(d), reps=60) - bench.empty_cost()
        print(f"{load} sliding: main {tm:.2f} us, csl8 {tn:.2f} us", flush=True)

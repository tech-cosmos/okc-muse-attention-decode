"""Native batch-1 decode-full kernel (tools/s73/cfl.cu): correctness + graph timing vs main.
python tools/s73/cfl.py [reps] [ng ...]   env NS (stages), ABL"""
import os, sys
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import torch, bench, reference, submission
from torch.utils.cpp_extension import load_inline
here = os.path.dirname(os.path.abspath(__file__))
NS = int(os.environ.get("NS", "3")); ABL = int(os.environ.get("ABL", "0"))
mod = load_inline("cfl_s73_%d_%d" % (NS, ABL), cpp_sources="void cfl_decode(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                  "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, double, int64_t);",
                  cuda_sources=open(os.path.join(here, "cfl.cu")).read(), functions=["cfl_decode"],
                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-DNS_=%d" % NS, "-DABL=%d" % ABL])
gpart = torch.empty(16 * 64 * 66, dtype=torch.float32, device="cuda")
cnt = torch.zeros(16, dtype=torch.int32, device="cuda")


def run(data, ng=16):
    cfg = data["config"]
    qkv = data["qkv"]
    out = torch.empty((cfg["batch"], 1, cfg["num_q_heads"] * 64), dtype=torch.bfloat16, device=qkv.device)
    mod.cfl_decode(qkv, data["k_cache"], data["v_cache"], data["page_table"], data["seq_lens"], data["cache_loc"],
                   data["k_scale"], data["v_scale"], out, gpart, cnt, cfg["page_size"],
                   cfg["sm_scale"] * cfg["qk_scale_factor"] * submission.LOG2E, ng)
    return out


ngs = [int(a) for a in sys.argv[2:]] or [16]
cases = [dict(batch=1, ctx=10, heads="tp8", layer="full"),
         dict(batch=1, ctx=127, heads="tp8", layer="full", page=1),
         dict(batch=1, ctx=3000, heads="tp8", layer="full"),
         dict(batch=1, ctx=6000, heads="tp8", layer="full", dist="edges", page=1),
         dict(batch=1, ctx=9000, heads="tp8", layer="full", kv_scale=0.015625),
         dict(batch=1, ctx=80000, heads="tp8", layer="full", dist="soft"),
         dict(batch=1, ctx=200000, heads="tp8", layer="full", page=1, dist="coherent"),
         dict(batch=1, ctx=80000, heads="tp8", layer="full", page=1, dist="coherent")]
if ABL == 0 and not os.environ.get("NOCHK"):
    for ng in ngs:
        for i, c in enumerate(cases):
            data = reference.generate_input(seed=7 + i, **c)
            for _ in range(2):
                out = run(data, ng)
            torch.cuda.synchronize()
            ok, msg = reference.check_implementation(data, reference.collect(data, out))
            print("PASS" if ok else "FAIL", ng, c, msg[:100], flush=True)
    assert int(cnt.abs().sum()) == 0, cnt

reps = int(sys.argv[1]) if len(sys.argv) > 1 else 3
for load, c in bench.SCORED:
    if c["layer"] != "full" or c["batch"] != 1 or c["q_len"] != 1:
        continue
    d = bench.data_for(load, c)
    for _ in range(reps):
        tm = bench.time_graph(lambda: submission.kernel_fn(d), reps=60) - bench.empty_cost()
        ts = [bench.time_graph(lambda: run(d, ng), reps=60) - bench.empty_cost() for ng in ngs]
        print(f"{load} full: main {tm:.2f} us, cfl " + " ".join(f"ng{ng} {t:.2f}" for ng, t in zip(ngs, ts)), flush=True)

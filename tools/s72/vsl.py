"""Native cluster verify-sliding kernel (tools/s72/csv.cu): correctness + paired timing vs main. python tools/s72/vsl.py rounds"""
import os, sys
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import torch, bench, reference, submission
from torch.utils.cpp_extension import load_inline
here = os.path.dirname(os.path.abspath(__file__))
mod = load_inline("csv_s72" + ("abl" if os.environ.get("CSV_SRC") else ""), cpp_sources="void csv_verify(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                  "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, double" + (", int64_t" if os.environ.get("CSV_SRC") else "") + ");",
                  cuda_sources=open(os.environ.get("CSV_SRC", os.path.join(here, "csv.cu"))).read(), functions=["csv_verify"],
                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"])


def run(data, abl=0):
    cfg = data["config"]
    qkv = data["qkv"]
    out = torch.empty((cfg["batch"], 8, cfg["num_q_heads"] * 64), dtype=torch.bfloat16, device=qkv.device)
    mod.csv_verify(qkv, submission._cos_sin(cfg["rope_theta"], 64, qkv.device), data["k_cache"], data["v_cache"],
                   data["page_table"], data["seq_lens"], data["cache_loc"], data["k_scale"], data["v_scale"], out,
                   cfg["page_size"], cfg["sm_scale"] * cfg["qk_scale_factor"] * submission.LOG2E, *((abl,) if os.environ.get("CSV_SRC") else ()))
    return out


cases = [dict(batch=1, ctx=10, heads="tp8", layer="sliding", q_len=8),
         dict(batch=1, ctx=3000, heads="tp8", layer="sliding", q_len=8, dist="edges", page=1),
         dict(batch=3, ctx=4000, heads="tp8", layer="sliding", q_len=8, page=1, kv_scale=0.015625),
         dict(batch=5, ctx=5000, heads="tp8", layer="sliding", q_len=8, spread="mixed", page=1),
         dict(batch=4, ctx=6000, heads="tp8", layer="sliding", q_len=8, dist="soft", page=1),
         dict(batch=2, ctx=2050, heads="tp8", layer="sliding", q_len=8, dist="edges"),
         dict(batch=1, ctx=80000, heads="tp8", layer="sliding", q_len=8, page=1, dist="soft")]
for i, c in enumerate(cases):
    data = reference.generate_input(seed=11 + i, **c)
    out = run(data)
    torch.cuda.synchronize()
    ok, msg = reference.check_implementation(data, reference.collect(data, out))
    print("PASS" if ok else "FAIL", c, msg[:100], flush=True)
rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
for load, c in bench.SCORED:
    if c["layer"] != "sliding" or c["q_len"] != 8 or c["batch"] > 16:
        continue
    d = bench.data_for(load, c)
    a, n = [], []
    for _ in range(rounds):
        a.append(bench.time_graph(lambda: submission.kernel_fn(d), reps=60) - bench.empty_cost())
        n.append(bench.time_graph(lambda: run(d), reps=60) - bench.empty_cost())
    wins = sum(y < x for x, y in zip(a, n))
    if os.environ.get("CSV_SRC"):
        print("  chain-only", min(bench.time_graph(lambda: run(d, 3), reps=60) - bench.empty_cost() for _ in range(3)),
              "no-merge", min(bench.time_graph(lambda: run(d, 1), reps=60) - bench.empty_cost() for _ in range(3)))
    print(f"{load} sliding: main {sorted(a)[len(a)//2]:.2f}  csv {sorted(n)[len(n)//2]:.2f}  csv wins {wins}/{rounds}", flush=True)

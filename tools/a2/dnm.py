"""agent2 s7: dnm (no-split decode sliding, tools/a2/dnm.cu): correctness vs the reference and H100 timing vs main.
  python tools/a2/dnm.py [variants]"""
import sys, torch
sys.path.insert(0, "/home/daytona/repo2/tools"); sys.path.insert(0, "/home/daytona/repo2")
import bench, reference, submission
from torch.utils.cpp_extension import load_inline
import os
DEF = os.environ.get("DNM_DEF", "")
m = load_inline(name="a2_dnm_" + str(abs(hash(open("/home/daytona/repo2/tools/a2/dnm.cu").read() + DEF)) % 10**8),
                cpp_sources="void dnm_decode(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, double, int64_t);",
                cuda_sources=open("/home/daytona/repo2/tools/a2/dnm.cu").read(), functions=["dnm_decode"],
                extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"] + DEF.split(), verbose=False)
variants = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 1, 2, 3, 4]
def run(data, v):
    cfg = data["config"]; d = cfg["head_dim"]
    out = torch.empty((cfg["batch"], 1, cfg["num_q_heads"] * d), dtype=torch.bfloat16, device="cuda")
    m.dnm_decode(data["qkv"], submission._cos_sin(cfg["rope_theta"], d, "cuda"), data["k_cache"], data["v_cache"],
                 data["page_table"], data["seq_lens"], data["cache_loc"], data["k_scale"], data["v_scale"], out,
                 cfg["page_size"], cfg["sm_scale"] * cfg["qk_scale_factor"] * submission.LOG2E, v)
    return out
cases = [dict(batch=32, ctx=80000, heads="tp8", layer="sliding", page=1, dist="soft", spread="agentic"),
         dict(batch=16, ctx=80000, heads="tp8", layer="sliding", page=1, dist="soft", spread="agentic"),
         dict(batch=1, ctx=10, heads="tp8", layer="sliding"), dict(batch=3, ctx=2047, heads="tp8", layer="sliding"),
         dict(batch=4, ctx=6000, heads="tp8", layer="sliding", dist="edges", page=1),
         dict(batch=40, ctx=1800, heads="tp8", layer="sliding", page=1, dist="edges", spread="mixed"),
         dict(batch=33, ctx=5000, heads="tp8", layer="sliding", dist="soft", kv_scale=0.015625, spread="mixed"),
         dict(batch=20, ctx=3000, heads="tp8", layer="sliding", page=1, dist="coherent", spread="mixed")]
for v in variants:
    for i, c in enumerate(cases):
        data = reference.generate_input(seed=301 + i, **c)
        out = run(data, v); torch.cuda.synchronize()
        ok, msg = reference.check_implementation(data, reference.collect(data, out))
        if not ok: print("FAIL", v, c, msg[:200], flush=True)
    print("variant", v, "checked", flush=True)
for load, c in bench.SCORED:
    if c["layer"] != "sliding" or c["q_len"] != 1 or c["batch"] == 1: continue
    data = bench.data_for(load, c)
    e = bench.empty_cost()
    base = bench.time_graph(lambda: submission.kernel_fn(data)) - e
    res = [bench.time_graph(lambda: run(data, v)) - e for v in variants]
    print(f"{load}: main {base:.2f} | " + " ".join(f"v{v} {x:.2f}" for v, x in zip(variants, res)), flush=True)

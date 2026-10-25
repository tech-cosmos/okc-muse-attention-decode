"""s96: verify sliding as csl-style clusters per (token, KV head) (vtk.cu) vs submission.kernel_fn.
  python tools/s96/vtk.py [check|time]"""
import os, sys, torch
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench, reference, submission
from torch.utils.cpp_extension import load_inline
SRC = os.environ.get("SRC", "/home/daytona/repo/tools/s96/vtk.cu")
m = load_inline(name="s96_vtk_" + os.path.basename(SRC).split(".")[0] + "_" + os.environ.get("ABL", "0"), cpp_sources="void vtk_verify(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, double, int64_t, int64_t);",
                cuda_sources=open(SRC).read(), functions=["vtk_verify"],
                extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3", "-DABL=" + os.environ.get("ABL", "0")], verbose=False)

def run(data, spec=1, ntok=0):
    cfg = data["config"]
    out = torch.empty((cfg["batch"], cfg["q_len"], cfg["num_q_heads"] * cfg["head_dim"]), dtype=torch.bfloat16, device="cuda")
    m.vtk_verify(data["qkv"], submission._cos_sin(cfg["rope_theta"], 64, "cuda"), data["k_cache"], data["v_cache"],
                 data["page_table"], data["seq_lens"], data["cache_loc"], data["k_scale"], data["v_scale"], out,
                 cfg["page_size"], cfg["sm_scale"] * cfg["qk_scale_factor"] * submission.LOG2E, spec, ntok)
    return out

cases = [dict(batch=1, ctx=80000, heads="tp8", layer="sliding", page=1, q_len=8, dist="soft"),
         dict(batch=5, ctx=5000, heads="tp8", layer="sliding", q_len=8, spread="mixed", page=1),
         dict(batch=3, ctx=4000, heads="tp8", layer="sliding", q_len=8, page=1, kv_scale=0.015625),
         dict(batch=4, ctx=6000, heads="tp8", layer="sliding", q_len=8, dist="soft", page=1),
         dict(batch=2, ctx=1000, heads="tp8", layer="sliding", q_len=8, page=64),
         dict(batch=1, ctx=10, heads="tp8", layer="sliding", q_len=8, dist="edges"),
         dict(batch=4, ctx=6000, heads="tp8", layer="sliding", q_len=8, dist="edges", page=1)]
mode = sys.argv[1] if len(sys.argv) > 1 else "check"
if mode == "check":
    for i, c in enumerate(cases):
        data = reference.generate_input(seed=4242 + i, **c)
        kc0, vc0 = data["k_cache"].clone(), data["v_cache"].clone()
        ref = submission.kernel_fn(data).clone()
        kc1, vc1 = data["k_cache"].clone(), data["v_cache"].clone()
        data["k_cache"].copy_(kc0); data["v_cache"].copy_(vc0)
        out = run(data)
        torch.cuda.synchronize()
        ok, msg = reference.check_implementation(data, reference.collect(data, out))
        d = (out.float() - ref.float()).norm() / ref.float().norm()
        same = torch.equal(kc1, data["k_cache"]) and torch.equal(vc1, data["v_cache"])
        print(f"{'PASS' if ok else 'FAIL'} {c} vs-main {d:.4f} cache-same {same} {msg[:100]}", flush=True)
else:
    c = cases[0]
    data = bench.data_for("verify 1", c)
    dd = bench.data_for("decode 1", dict(batch=1, ctx=80000, heads="tp8", layer="sliding", page=1, dist="soft"))
    for r in range(4):
        print(f"decode 1 csl {bench.time_graph(lambda: submission.kernel_fn(dd)) - bench.empty_cost():.2f}")
        a = bench.time_graph(lambda: submission.kernel_fn(data)) - bench.empty_cost()
        bs = [bench.time_graph(lambda: run(data, 1, nt)) - bench.empty_cost() for nt in (0, 1, 2, 4)]
        print(f"round {r}: main {a:.2f} us, vtk all / 1 / 2 / 4 tokens " + " ".join(f"{x:.2f}" for x in bs), flush=True)

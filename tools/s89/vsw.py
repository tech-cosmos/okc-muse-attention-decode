"""s89: native wgmma verify sliding kernel (vsw.cu) vs submission.kernel_fn: correctness and graph timing.
  python tools/s84/vsl.py [check|time]"""
import os, sys, statistics, torch
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench, reference, submission
from torch.utils.cpp_extension import load_inline
m = load_inline(name="s89_vsw7_" + os.environ.get("BULK", "1"), cpp_sources="void vsw_dbg(torch::Tensor); void vsw_verify(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, double, int64_t);",
                cuda_sources=open("/home/daytona/repo/tools/s89/vsw.cu").read(), functions=["vsw_verify", "vsw_dbg"],
                extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3", "-DVSW_BULK=" + os.environ.get("BULK", "1")], verbose=False)

def run(data, abl=0):
    cfg = data["config"]
    out = torch.empty((cfg["batch"], cfg["q_len"], cfg["num_q_heads"] * cfg["head_dim"]), dtype=torch.bfloat16, device="cuda")
    m.vsw_verify(data["qkv"], submission._cos_sin(cfg["rope_theta"], 64, "cuda"), data["k_cache"], data["v_cache"],
                 data["page_table"], data["seq_lens"], data["cache_loc"], data["k_scale"], data["v_scale"], out,
                 cfg["page_size"], cfg["sm_scale"] * cfg["qk_scale_factor"] * submission.LOG2E, abl)
    return out

cases = [dict(batch=1, ctx=80000, heads="tp8", layer="sliding", page=1, q_len=8, dist="soft"),
         dict(batch=5, ctx=5000, heads="tp8", layer="sliding", q_len=8, spread="mixed", page=1),
         dict(batch=3, ctx=4000, heads="tp8", layer="sliding", q_len=8, page=1, kv_scale=0.015625),
         dict(batch=4, ctx=6000, heads="tp8", layer="sliding", q_len=8, dist="soft", page=1),
         dict(batch=2, ctx=1000, heads="tp8", layer="sliding", q_len=8, page=64),
         dict(batch=1, ctx=10, heads="tp8", layer="sliding", q_len=8, dist="edges"),
         dict(batch=4, ctx=6000, heads="tp8", layer="sliding", q_len=8, dist="edges", page=1)]
mode = sys.argv[1] if len(sys.argv) > 1 else "check"
X = int(sys.argv[2]) if len(sys.argv) > 2 else 0
if mode == "check":
    for i, c in enumerate(cases):
        data = reference.generate_input(seed=4242 + i, **c)
        ref = submission.kernel_fn(data).clone()
        kc0, vc0 = data["k_cache"].clone(), data["v_cache"].clone()
        out = run(data, X)
        torch.cuda.synchronize()
        ok, msg = reference.check_implementation(data, reference.collect(data, out))
        d = (out.float() - ref.float()).norm() / ref.float().norm()
        same = torch.equal(kc0, data["k_cache"]) and torch.equal(vc0, data["v_cache"])
        print(f"{'PASS' if ok else 'FAIL'} {c} vs-main {d:.4f} cache-same {same} {msg[:100]}", flush=True)
else:
    c = cases[0]
    data = bench.data_for("verify 1", c)
    for r in range(3):
        a = bench.time_graph(lambda: submission.kernel_fn(data)) - bench.empty_cost()
        bs = [bench.time_graph(lambda: run(data, ab)) - bench.empty_cost() for ab in (0, 1, 2, 4, 3, 10)]
        print(f"round {r}: main {a:.2f} us, vsw full/loads/nopush/straggler/nomerge/nospec " + " ".join(f"{x:.2f}" for x in bs), flush=True)


if mode == "stamp":
    import numpy as np
    c = cases[0]
    data = bench.data_for("verify 1", c)
    flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda"); sink = torch.zeros((), dtype=torch.int64, device="cuda")
    res = []
    for r in range(9):
        torch.sum(flush.view(torch.int64), 0, out=sink)
        run(data, 9)
        torch.cuda.synchronize()
        d = torch.zeros(32 * 3 * 8, dtype=torch.int64); m.vsw_dbg(d)
        a = d.view(32, 3, 8).numpy().astype(np.float64)
        t0 = a[:, 0, 0].min()
        res.append((a - t0) / 1000.0)
    a = np.median(np.stack(res), 0)
    names = ["start", "loads+STS", "bar", "QK done", "PV done", "merged", "push", "owner done"]
    for k, nm in enumerate(names):
        for wg in range(3):
            v = a[:, wg, k] if wg < 2 else a[[15, 31], 2, k]
            if wg > 0 and k >= 5: continue
            print(f"{nm:11s} wg{wg} min {v.min():5.2f} med {np.median(v):5.2f} max {v.max():5.2f}")

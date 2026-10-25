"""s92: native decode-1-full kernel (tools/s92/nfl.cu) + submission's _combine vs submission.kernel_fn.
NW=4 DEP=3 NCTA=128 python tools/s92/nfl.py [check|time] [rounds]"""
import os, sys, statistics
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import torch, triton, bench, reference, submission
from torch.utils.cpp_extension import load_inline

NW, DEP, NCTA = int(os.environ.get("NW", 4)), int(os.environ.get("DEP", 3)), int(os.environ.get("NCTA", 128))
src = open(os.environ.get("NFL_SRC", os.path.join(os.path.dirname(__file__), "nfl.cu"))).read()
mod = load_inline(name=f"nfl_{NW}_{DEP}_{abs(hash(src)) % 10**8}", cpp_sources="void nfl_decode(torch::Tensor, torch::Tensor, "
                  "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                  "torch::Tensor, int64_t, double);", cuda_sources=src, functions=["nfl_decode"],
                  extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3", f"-DNFL_NW={NW}", f"-DNFL_DEP={DEP}",
                                     "-Xptxas=-v"] + (["-lineinfo"] if os.environ.get("LI") else []),
                  verbose=bool(os.environ.get("V")))


def nfl_fn(data):
    cfg = data["config"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    qkv = data["qkv"]
    pacc = torch.empty((1, hkv, NCTA, 8, d), dtype=torch.float32, device=qkv.device)
    pml = torch.empty((1, hkv, NCTA, 8, 2), dtype=torch.float32, device=qkv.device)
    out = torch.empty((1, 1, hq * d), dtype=torch.bfloat16, device=qkv.device)
    mod.nfl_decode(qkv, data["k_cache"], data["v_cache"], data["page_table"], data["seq_lens"], data["cache_loc"],
                   data["k_scale"], data["v_scale"], pacc, pml, NCTA, cfg["sm_scale"] * cfg["qk_scale_factor"] * submission.LOG2E)
    sp = triton.next_power_of_2(NCTA)
    submission._combine[(1, hq)](qkv, pacc, pml, data["v_scale"], out, HQ=hq, HKV=hkv, G=8, L=1, M=8, D=d, S=NCTA, S_PAD=sp,
                                 SB=min(128, sp), ONE=submission.MERGE1, PDL=True, num_warps=4 if NCTA > 32 else 2,
                                 launch_pdl=True)
    return out


CASES = [dict(batch=1, ctx=80000, heads="tp8", layer="full", page=1, q_len=1, dist="coherent"),
         dict(batch=1, ctx=10, heads="tp8", layer="full", page=1, q_len=1),
         dict(batch=1, ctx=3000, heads="tp8", layer="full", page=1, q_len=1, dist="edges"),
         dict(batch=1, ctx=100000, heads="tp8", layer="full", page=1, q_len=1, kv_scale=0.015625),
         dict(batch=1, ctx=4097, heads="tp8", layer="full", page=1, q_len=1, dist="soft")]

mode = sys.argv[1] if len(sys.argv) > 1 else "check"
if mode == "check":
    for i, c in enumerate(CASES):
        data = reference.generate_input(seed=77 + i, **c)
        out = nfl_fn(data)
        torch.cuda.synchronize()
        ok, msg = reference.check_implementation(data, reference.collect(data, out))
        print("PASS" if ok else "FAIL", c, msg[:100], flush=True)
elif mode == "time":
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    data = reference.generate_input(seed=99, **CASES[0])
    e = bench.empty_cost()
    a, b = [], []
    for r in range(rounds):
        a.append(bench.time_graph(lambda: submission.kernel_fn(data)) - e)
        b.append(bench.time_graph(lambda: nfl_fn(data)) - e)
    print(f"NW {NW} DEP {DEP} NCTA {NCTA}: main {statistics.median(a):.2f}  nfl {statistics.median(b):.2f}  "
          f"wins {sum(y < x for x, y in zip(a, b))}/{rounds}", flush=True)
if mode == "parts":
    data = reference.generate_input(seed=99, **dict(CASES[0], ctx=int(os.environ.get("CTX", 80000))))
    cfg = data["config"]; qkv = data["qkv"]
    pacc = torch.empty((1, 2, NCTA, 8, 64), dtype=torch.float32, device="cuda"); pml = torch.empty((1, 2, NCTA, 8, 2), device="cuda")
    k_only = lambda: mod.nfl_decode(qkv, data["k_cache"], data["v_cache"], data["page_table"], data["seq_lens"], data["cache_loc"],
                   data["k_scale"], data["v_scale"], pacc, pml, NCTA, 0.1)
    e = bench.empty_cost()
    cold = statistics.median([bench.time_graph(k_only) - e for _ in range(3)])
    bench.OLD = False
    import time
    g = torch.cuda.CUDAGraph()
    k_only(); torch.cuda.synchronize()
    with torch.cuda.graph(g): k_only()
    ts = []
    for _ in range(50):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        g.replay(); a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1000)
    print(f"NW {NW} DEP {DEP} NCTA {NCTA}: kernel cold {cold:.2f}  warm(incl launch) {statistics.median(ts):.2f} empty {e:.2f}", flush=True)

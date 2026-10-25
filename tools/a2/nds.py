"""agent2 s10: native decode-sliding kernel nds (tools/a2/nds.cu): correctness vs the reference + timing vs main.
  NDS_S=4,8 NDS_DEF='-DNV_NS=4 ...' python tools/a2/nds.py [check] [time]"""
import sys, os, torch, triton
sys.path.insert(0, "/home/daytona/repo2/tools"); sys.path.insert(0, "/home/daytona/repo2")
import bench, reference, submission
from torch.utils.cpp_extension import load_inline
defs = os.environ.get("NDS_DEF", "").split()
src = open("/home/daytona/repo2/tools/a2/nds.cu").read()
mod = load_inline(name=f"a2nds_{abs(hash(src + str(defs))) % 10**10}",
    cpp_sources="void nds(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t, int64_t, int64_t);",
    cuda_sources=src, functions=["nds"], extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"] + defs)
CL = any("NDS_CL" in x and not x.endswith("=0") for x in defs)
HB = int(os.environ.get('NDS_HB', '2')); CW = int(os.environ.get('NDS_CW', '1'))
SS = [int(x) for x in os.environ.get("NDS_S", "4").split(",")]

def run(data, S, combine=True):
    cfg = data["config"]; B, hq, hkv, d = cfg["batch"], 16, 2, 64
    qkv = data["qkv"]; dev = qkv.device
    ch = triton.cdiv(triton.cdiv(cfg["window"], S), 64) * 64
    pacc = torch.empty((B, hkv, S, 8, d), dtype=torch.float32, device=dev)
    pml = torch.empty((B, hkv, S, 8, 2), dtype=torch.float32, device=dev)
    out = torch.empty((B, 1, hq * d), dtype=torch.bfloat16, device=dev)
    mod.nds(qkv, submission._cos_sin(cfg["rope_theta"], d, dev), data["k_cache"], data["v_cache"], data["k_scale"],
            data["v_scale"], data["cache_loc"], data["page_table"], data["seq_lens"], pacc, pml, out,
            cfg["sm_scale"] * cfg["qk_scale_factor"] * submission.LOG2E, S, ch, cfg["window"])
    if combine and not CL:
        submission._combine_hb[(B, hq // HB)](qkv, pacc, pml, data["v_scale"], out, HQ=hq, HKV=hkv, G=8, L=1, M=8, D=d,
            S=S, S_PAD=triton.next_power_of_2(S), SB=S, HB=HB, ONE=submission.MERGE1, PDL=submission.USE_PDL,
            num_warps=CW, launch_pdl=submission.USE_PDL)
    return out

if "check" in sys.argv:
    cases = [dict(batch=32, ctx=80000, heads="tp8", layer="sliding", page=1, dist="soft", spread="agentic"),
             dict(batch=16, ctx=80000, heads="tp8", layer="sliding", page=1, dist="soft", spread="agentic"),
             dict(batch=20, ctx=1800, heads="tp8", layer="sliding", page=1, dist="edges", spread="mixed"),
             dict(batch=32, ctx=1500, heads="tp8", layer="sliding", page=1, dist="edges", spread="mixed"),
             dict(batch=40, ctx=5000, heads="tp8", layer="sliding", page=1, dist="soft", kv_scale=0.015625, spread="mixed"),
             dict(batch=33, ctx=3000, heads="tp8", layer="sliding", page=1, dist="coherent", spread="mixed"),
             dict(batch=3, ctx=10, heads="tp8", layer="sliding", page=1),
             dict(batch=64, ctx=20000, heads="tp8", layer="sliding", page=1, dist="edges", spread="agentic")]
    for S in SS:
        for i, c in enumerate(cases):
            data = reference.generate_input(seed=301 + i, **c)
            out = run(data, S); torch.cuda.synchronize()
            ok, msg = reference.check_implementation(data, reference.collect(data, out))
            print("PASS" if ok else "FAIL", S, c, msg[:150], flush=True)
if "time" in sys.argv:
    e = bench.empty_cost()
    for load, c in bench.SCORED:
        if c["layer"] != "sliding" or c["q_len"] != 1 or c["batch"] == 1: continue
        data = bench.data_for(load, c)
        for r in range(int(os.environ.get("NDS_R", "3"))):
            res = [f"main {bench.time_graph(lambda: submission.kernel_fn(data)) - e:.2f}"]
            for S in SS:
                res.append(f"S{S} {bench.time_graph(lambda: run(data, S)) - e:.2f}")
                res.append(f"k{S} {bench.time_graph(lambda: run(data, S, False)) - e:.2f}")
            print(load, " | ".join(res), flush=True)

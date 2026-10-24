"""Graph-timed, cold-L2 sliding gather floor (tools/s69/sgather.cu) on the scored shapes' data.
python tools/s69/sgather.py"""
import os, sys, fcntl
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import torch, bench
from torch.utils.cpp_extension import load_inline
here = os.path.dirname(os.path.abspath(__file__))
mod = load_inline("sgather_s69", cpp_sources="void run(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int, int, int, torch::Tensor);",
                  cuda_sources=open(os.path.join(here, "sgather.cu")).read(), functions=["run"], extra_cuda_cflags=["-O3"])
out = torch.zeros(4096, device="cuda")
empty = bench.time_graph(lambda: None) if False else None
for ld, c in bench.SCORED:
    if c["layer"] != "sliding" or c["q_len"] != 1: continue
    d = bench.data_for(ld, c)
    kc, vc = d["k_cache"].view(torch.uint8), d["v_cache"].view(torch.uint8)
    kc = kc.reshape(kc.shape[0], -1); vc = vc.reshape(vc.shape[0], -1)
    pt = d["page_table"].to(torch.int32); seq = d["seq_lens"].to(torch.int32)
    res = []
    for kpc, th in ((512, 512), (256, 256), (256, 512), (128, 256), (128, 128), (64, 128)):
        t = bench.time_graph(lambda: mod.run(kc, vc, pt, seq, kpc, 2048, th, out))
        res.append(f"kpc {kpc} th {th}: {t - bench.empty_cost():.2f}")
    print(ld, c["batch"], " | ".join(res), flush=True)

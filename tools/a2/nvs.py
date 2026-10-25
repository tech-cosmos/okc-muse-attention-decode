"""agent2: build + check + time the native verify-sliding prototype tools/a2/nvs.cu.
  python tools/a2/nvs.py [check|time] [S] (env NV_NS stages, NV_LAG, NV_PFD, NV_PREGS/NV_CREGS)"""
import os, sys, statistics, torch
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0a")
sys.path.insert(0, "/home/daytona/repo2/tools"); sys.path.insert(0, "/home/daytona/repo2")
from torch.utils.cpp_extension import load_inline
import bench, submission
if os.environ.get("NONEW"):
    # Reference without the new tokens (split 0 then matches nvs): load a patched copy of submission.py.
    import importlib
    src = open("/home/daytona/repo2/submission.py").read().replace("    if split == (S if SEP else 0):\n        # The block", "    if split == (S + 7 if SEP else -1):\n        # The block")
    open("/tmp/a2nonew_sub.py", "w").write(src); sys.path.insert(0, "/tmp")
    submission = importlib.import_module("a2nonew_sub"); bench.submission = submission
here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
E = {k: int(os.environ.get(k, v)) for k, v in dict(NV_NS=3, NV_LAG=1, NV_PFD=4, NV_PREGS=40, NV_CREGS=96, NV_LAZY=1, NV_STAMP=0, NV_ABL=0, NV_L2PF=0, NV_NEW=1, NV_MINB=2, NV_SEP=0).items()}
tag = "_".join(f"{v}" for v in E.values())
NVF = os.environ.get("NVS_FILE", "nvs.cu")
CLEAN = NVF != "nvs.cu"
mod = load_inline(f"a2nvs_{tag}_{abs(hash(open(os.path.join(here, NVF)).read())) % 10**8}",
    cpp_sources="void nvs(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t, int64_t, int64_t" + ("" if CLEAN else ", torch::Tensor") + ");",
    cuda_sources=open(os.path.join(here, NVF)).read(), functions=["nvs"],
    extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas=-v"] + [f"-D{k}={v}" for k, v in E.items()],
    verbose=False)
mode = sys.argv[1] if len(sys.argv) > 1 else "check"
S = int(sys.argv[2]) if len(sys.argv) > 2 else 4
load = sys.argv[3] if len(sys.argv) > 3 else "verify 32"
for l, c in bench.SCORED:
    if l == load and c["layer"] == "sliding":
        data, cfg = bench.data_for(l, c), None
cfg = data["config"]
B, L, hq, hkv, d = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
cs = submission._cos_sin(cfg["rope_theta"], d, "cuda")
scale = cfg["sm_scale"] * cfg["qk_scale_factor"] * submission.LOG2E
ch = triton_ch = -(-(-(-2048 // S)) // 64) * 64
SP = S + E["NV_SEP"]
pacc = torch.empty((B, hkv, SP, L * 8, d), dtype=torch.float32, device="cuda")
pml = torch.empty((B, hkv, SP, L * 8, 2), dtype=torch.float32, device="cuda")
out = torch.empty((B, L, hq * d), dtype=torch.bfloat16, device="cuda")
stb = torch.zeros(B * S * 2 * 8, dtype=torch.int64, device="cuda")
def run_nvs():
    mod.nvs(data["qkv"], cs, data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"], data["cache_loc"], data["page_table"], data["seq_lens"],
            pacc, pml, scale, S, ch, cfg["window"], *([] if CLEAN else [stb]))
def run_merge_nt():
    import cnt
    cnt._combine_nt[(B * L, hq // 2)](data["qkv"], cs, data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"],
        data["seq_lens"], data["cache_loc"], pacc, pml, out, scale, HQ=hq, HKV=hkv, G=8, L=L, M=L * 8, D=d, S=S,
        S_PAD=triton_next(S), HB=2, PDL=True, num_warps=1, launch_pdl=True)
if not E["NV_NEW"]:
    run_merge = run_merge_nt
else:
  def run_merge():
    submission._combine_hb[(B * L, hq // 2)](data["qkv"], pacc, pml, data["v_scale"], out, HQ=hq, HKV=hkv, G=8, L=L,
        M=L * 8, D=d, S=SP, S_PAD=triton_next(SP), SB=triton_next(SP), HB=2, ONE=True, PDL=True, num_warps=1, launch_pdl=True)
def triton_next(x): return 1 << (x - 1).bit_length()
if mode == "check":
    # Triton's partials (same S, block_n 64) for reference; split 0 there also holds the new tokens.
    got = {}
    orig = submission._combine_hb
    class Cap:
        def __getitem__(self, g):
            def f(*a, **k): got["pacc"], got["pml"] = a[1].clone(), a[2].clone()
            return f
    submission._combine_hb = Cap()
    po = submission._plan
    submission._plan = lambda cf, mp, dv: {**po(cf, mp, dv), "splits": S, "sb": S, "block_n": 64, "tsep": False}
    submission.kernel_fn(data)
    submission._combine_hb, submission._plan = orig, po
    run_nvs(); torch.cuda.synchronize()
    ta, tm = got["pacc"], got["pml"]
    for sp in range(S):
        o1 = pacc[:, :, sp] / pml[:, :, sp, :, 1:2]; o2 = ta[:, :, sp] / tm[:, :, sp, :, 1:2]
        lse1 = pml[:, :, sp, :, 0] + torch.log2(pml[:, :, sp, :, 1]); lse2 = tm[:, :, sp, :, 0] + torch.log2(tm[:, :, sp, :, 1])
        print(f"split {sp}: o max abs diff {(o1 - o2).abs().max().item():.4f} (|o| max {o2.abs().max().item():.3f}), lse diff {(lse1 - lse2).abs().max().item():.4f}")
elif mode == "out":
    import reference
    ref = submission.kernel_fn(data).clone()
    run_nvs(); run_merge(); torch.cuda.synchronize()
    print("max |nvs - triton|", (out.float() - ref.float()).abs().max().item(), " max |triton|", ref.float().abs().max().item())
    ok, msg = reference.check_implementation(data, reference.collect(data, out))
    print("reference check:", ok, msg[:200])
    ok, msg = reference.check_implementation(data, reference.collect(data, ref))
    print("reference check (triton):", ok, msg[:200])
elif mode == "stamp":
    fl = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
    for _ in range(3):
        fl.zero_(); run_nvs()
    torch.cuda.synchronize()
    t = stb.view(-1, 8).cpu().double()
    z = t[:, 0].min()
    for i, nm in enumerate(["start", "q ready", "tile0 ready", "loop done", "stored"]):
        x = (t[:, i] - z) / 1e3
        last = (torch.arange(len(x)) // 2) % S == S - 1
        print(f"{nm:12s} min {x.min():6.2f} med {x.median():6.2f} max {x.max():6.2f} | last split med {x[last].median():6.2f} max {x[last].max():6.2f}")
else:
    a = statistics.median(bench.time_graph(run_nvs) - bench.empty_cost() for _ in range(5))
    b = statistics.median(bench.time_graph(lambda: (run_nvs(), run_merge())) - bench.empty_cost() for _ in range(5))
    print(f"{load} nvs S {S} {E}: attn-only {a:.2f} us, + _combine_hb {b:.2f} us")
if mode == "merge":
    def mhb():
        submission._combine_hb[(B * L, hq // 2)](data["qkv"], pacc, pml, data["v_scale"], out, HQ=hq, HKV=hkv, G=8, L=L,
            M=L * 8, D=d, S=S, S_PAD=triton_next(S), SB=triton_next(S), HB=2, ONE=True, PDL=True, num_warps=1, launch_pdl=True)
    run_nvs()
    for nm, f in (("combine_hb", mhb), ("combine_nt", run_merge_nt)):
        print(nm, "alone", round(statistics.median(bench.time_graph(f) - bench.empty_cost() for _ in range(5)), 2))

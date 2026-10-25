"""s97: s71 nverify at verify 1 full size (attention only, cached keys) vs main's Triton attention-only. H100 timing."""
import sys, os, json; sys.path.insert(0, "tools"); sys.path.insert(0, "tools/s71"); sys.path.insert(0, ".")
import torch, submission, bench, build
load, c = [x for x in bench.SCORED if x[0] == "verify 1" and x[1]["layer"] == "full"][0]
import reference
c = {**c, "ctx": int(os.environ.get("CTX", "80000"))}
data = reference.generate_input(seed=99, **c)
cfg = data["config"]
NV = build.build(int(os.environ.get("NS", "3")))
qs = cfg["sm_scale"] * cfg["qk_scale_factor"] * 1.4426950408889634
class Nop:
    def __getitem__(self, g): return lambda *a, **k: None
orig = {k: getattr(submission, k) for k in ("_combine", "_new_tokens", "_combine_hb")}
def attn_main():
    for k in orig: setattr(submission, k, Nop())
    t = bench.time_graph(lambda: submission.kernel_fn(data)) - bench.empty_cost()
    for k, v in orig.items(): setattr(submission, k, v)
    return t
def nv(P):
    pacc = torch.empty((2, P + 1, 64, 64), dtype=torch.float32, device="cuda")
    pml = torch.empty((2, P + 1, 64, 2), dtype=torch.float32, device="cuda")
    f = lambda: NV.nverify(data["qkv"], data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"],
                           data["seq_lens"], pacc, pml, qs, P, P + 1, cfg["page_size"])
    return bench.time_graph(f) - bench.empty_cost()
Ps = [int(x) for x in os.environ.get("PS", "132,264,396").split(",")]
for r in range(3):
    print(r, "main attn %.2f" % attn_main(), " ".join("nv%d %.2f" % (P, nv(P)) for P in Ps), flush=True)

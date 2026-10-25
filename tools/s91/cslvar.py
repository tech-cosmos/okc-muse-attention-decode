"""s91: csl (decode sliding cluster kernel) source variants vs submission's: correctness + paired graph timing.
VAR=sig,tree,cw python tools/s91/cslvar.py [rounds]"""
import os, sys
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import torch, bench, reference, submission
from torch.utils.cpp_extension import load_inline

def patch(src, var):
    reps = []
    if "sig" in var:  # gate sigmoid before the receive wait; fast reciprocal/division in the owner merge
        reps += [("    if (warp == 0) {\n        asm volatile(\"{\\n\\t.reg .pred p;",
                  "    if (warp == 0) {\n        const float sg0 = __frcp_rn(1.f + __expf(-g0)), sg1 = __frcp_rn(1.f + __expf(-g1));\n"
                  "        asm volatile(\"{\\n\\t.reg .pred p;"),
                 ("        float r = vscale / den;", "        float r = __fdividef(vscale, den);"),
                 ("        o0 *= ni / (1.f + __expf(-g0));\n        o1 *= ni / (1.f + __expf(-g1));",
                  "        o0 *= ni * sg0;\n        o1 *= ni * sg1;")]
    if "tree" in var:  # owner merge: tree max, 4 independent accumulators
        reps += [("""        float M = -1e30f;
#pragma unroll
        for (int s = 0; s < 16; ++s) M = fmaxf(M, rml[s][0]);
        float den = 0.f, o0 = 0.f, o1 = 0.f;
#pragma unroll
        for (int s = 0; s < 16; ++s) {
            float w = exp2f(rml[s][0] - M);
            den += w * rml[s][1];
            float2 a = *(const float2*)&racc[s][2 * lane];
            o0 += w * a.x; o1 += w * a.y;
        }
""", """        float mm[16], ll[16];
        float2 aa[16];
#pragma unroll
        for (int s = 0; s < 16; ++s) { float2 t = *(const float2*)&rml[s][0]; mm[s] = t.x; ll[s] = t.y; aa[s] = *(const float2*)&racc[s][2 * lane]; }
        float t8[8], t4[4];
#pragma unroll
        for (int s = 0; s < 8; ++s) t8[s] = fmaxf(mm[s], mm[s + 8]);
#pragma unroll
        for (int s = 0; s < 4; ++s) t4[s] = fmaxf(t8[s], t8[s + 4]);
        const float M = fmaxf(fmaxf(t4[0], t4[2]), fmaxf(t4[1], t4[3]));
        float dd[4] = {0.f, 0.f, 0.f, 0.f}, p0[4] = {0.f, 0.f, 0.f, 0.f}, p1[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int s = 0; s < 16; ++s) {
            float w = exp2f(mm[s] - M);
            dd[s & 3] += w * ll[s]; p0[s & 3] += w * aa[s].x; p1[s & 3] += w * aa[s].y;
        }
        float den = (dd[0] + dd[1]) + (dd[2] + dd[3]), o0 = (p0[0] + p0[1]) + (p0[2] + p0[3]), o1 = (p1[0] + p1[1]) + (p1[2] + p1[3]);
""")]
    if "cw" in var:  # cluster wait right after QK (all peers arrived at their start)
        reps += [('        asm volatile("barrier.cluster.wait.aligned;" ::: "memory");\n', ""),
                 ("    const float scale = qk_scale * kscale;\n",
                  '    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");\n    const float scale = qk_scale * kscale;\n')]
    if "pmt" in var:  # premerge: tree-free but fast exp; nothing else
        reps += [("            float w = exp2f(mk[k] - M);\n            L += w * lk[k];", "            float w = exp2f(mk[k] - M);\n            L = fmaf(w, lk[k], L);")]
    if "cpk" in var:  # one 16-CTA cluster per (request, KV head): 32 CTAs x 4 warps; owners = ranks 0-7
        reps += [("__launch_bounds__(256) decode", "__launch_bounds__(128) decode"),
                 ("    const int h = warp >> 2, wk = warp & 3;", "    const int h = (blockIdx.x >> 4) & 1, wk = warp;"),
                 ("    const int b = blockIdx.x / 16;", "    const int b = blockIdx.x / 32;"),
                 ('    if (threadIdx.x == 0) {\n        asm volatile("mbarrier.init', '    if (threadIdx.x == 0 && rank < 8) {\n        asm volatile("mbarrier.init'),
                 ("(HQ + 2 * HKV) * D + rank * D + 2 * lane)", "(HQ + 2 * HKV) * D + (h * 8 + (rank & 7)) * D + 2 * lane)"),
                 ("        const int h2 = tid >> 7, row", "        const int h2 = h, row"),
                 ("        const unsigned owner = h2 * 8 + row;", "        const unsigned owner = row;"),
                 ("    if (warp == 0) {\n", "    if (warp == 0 && rank < 8) {\n"),
                 ("out + (long)b * HQ * D + rank * D + 2 * lane)", "out + (long)b * HQ * D + (h * 8 + rank) * D + 2 * lane)"),
                 ("    cfg.gridDim = dim3(16 * batch);\n    cfg.blockDim = dim3(256);", "    cfg.gridDim = dim3(32 * batch);\n    cfg.blockDim = dim3(128);")]
    for a, b in reps:
        assert src.count(a) == 1, a[:80]
        src = src.replace(a, b)
    return src

def build(src, name):
    return load_inline(name=name, cpp_sources="void csl_decode(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                       "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, double);",
                       cuda_sources=src, functions=["csl_decode"],
                       extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"], verbose=False)

if __name__ == "__main__":
    var = os.environ.get("VAR", "")
    import hashlib
    src = patch(submission._CSL_SRC, var)
    new = build(src, "s91v_" + hashlib.md5(src.encode()).hexdigest()[:10])
    base = submission._CSL
    cases = [dict(batch=1, ctx=10, heads="tp8", layer="sliding"),
             dict(batch=3, ctx=2047, heads="tp8", layer="sliding"),
             dict(batch=4, ctx=6000, heads="tp8", layer="sliding", dist="edges", page=1),
             dict(batch=1, ctx=6000, heads="tp8", layer="sliding", kv_scale=0.015625),
             dict(batch=1, ctx=80000, heads="tp8", layer="sliding", page=1, dist="soft")]
    for i, c in enumerate(cases):
        data = reference.generate_input(seed=7 + i, **c)
        submission._CSL = new
        out = submission.kernel_fn(data); torch.cuda.synchronize()
        ok, msg = reference.check_implementation(data, reference.collect(data, out))
        print("PASS" if ok else "FAIL", c, msg[:100], flush=True)
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    for load, c in bench.SCORED:
        if c["layer"] != "sliding" or c["q_len"] != 1 or c["batch"] > 16:
            continue
        d = bench.data_for(load, c)
        a, b = [], []
        for _ in range(rounds):
            for mod, lst in ((base, a), (new, b)):
                submission._CSL = mod
                lst.append(bench.time_graph(lambda: submission.kernel_fn(d), reps=60) - bench.empty_cost())
        wins = sum(y < x for x, y in zip(a, b))
        md = lambda l: sorted(l)[len(l) // 2]
        print(f"{load} sliding [{var}]: base {md(a):.3f}  var {md(b):.3f}  diff {md(b) - md(a):+.3f}  var wins {wins}/{rounds}", flush=True)

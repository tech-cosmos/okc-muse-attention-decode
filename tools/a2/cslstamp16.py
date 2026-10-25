# agent2 s22: tools/s93/cslstamp.py for decode 16 sliding (512 CTAs), percentiles.
"""s91: %globaltimer phase stamps inside submission's csl kernel (decode 1 sliding), cold L2, median over runs.
python tools/s91/cslstamp.py [runs]"""
import os, sys, fcntl
sys.path.insert(0, "/home/daytona/repo2/tools"); sys.path.insert(0, "/home/daytona/repo")
import numpy as np, torch, bench, submission
from torch.utils.cpp_extension import load_inline

sys.path.insert(0, '/home/daytona/repo2/tools/s93'); sys.path.insert(0, '/home/daytona/repo2/tools/s91')
import cslvar2
import cslvar, hashlib
src = cslvar.patch(submission._CSL_SRC, os.environ.get('VAR', ''))
ST = lambda k: f'asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(_t)); if (lane == 0) g_dbg[(blockIdx.x * 4 + warp) * 13 + {k}] = _t;\n'
src = src.replace("namespace csl {", "namespace csl {\n__device__ unsigned long long g_dbg[512 * 4 * 13];", 1)
rep = [
    ("    CSL_LOADS\n    if (spec >= 0) {", "    CSL_LOADS\n    { unsigned long long _d; asm volatile(\"add.s64 %0, %1, %2;\" : \"=l\"(_d) : \"l\"(kslot[3]), \"l\"(vslot[7])); if (_d == 1234567) g_dbg[0] = 0; }\n" + ST(1) + "    if (spec >= 0) {"),
    ("    const unsigned mb = (unsigned)__cvta_generic_to_shared(&mbar);",
     "    unsigned long long _t;\n" + ST(0) + "    const unsigned mb = (unsigned)__cvta_generic_to_shared(&mbar);"),
    ("        mma8(sc[j], q8[2], 0u, q8[3], 0u, kr[j].z, kr[j].w);\n    }",
     "        mma8(sc[j], q8[2], 0u, q8[3], 0u, kr[j].z, kr[j].w);\n    }\n" + ST(2)),
    ("    // Pre-merge the 4 warps", ST(3) + "    // Pre-merge the 4 warps"),
    ("    __syncthreads();\n    {\n        const int h2", "    __syncthreads();\n" + ST(4) + "    {\n        const int h2"),
    ('        asm volatile("barrier.cluster.wait.aligned;" ::: "memory");\n',
     ST(12) + '        asm volatile("barrier.cluster.wait.aligned;" ::: "memory");\n' + ST(5)),
    ("    // Merge query head `rank`", ST(6) + "    // Merge query head `rank`"),
    ("@!p bra W;\\n\\t}\"\n                     ::\"r\"(mb) : \"memory\");\n", "@!p bra W;\\n\\t}\"\n                     ::\"r\"(mb) : \"memory\");\n" + ST(7)),
    ("__floats2bfloat162_rn(o0, o1);\n    }\n}", "__floats2bfloat162_rn(o0, o1);\n" + ST(8) + "    }\n}"),
]
rep += [
    ("        float r = vscale / den;", ST(9) + "        float r = vscale / den;"),
    ("        float ni = rsqrtf(s2 / D + 1e-5f);", ST(10) + "        float ni = rsqrtf(s2 / D + 1e-5f);"),
    ("        *(__nv_bfloat162*)(out + (long)b * HQ * D", ST(11) + "        *(__nv_bfloat162*)(out + (long)b * HQ * D")]
i = src.index('// s96 vtk'); head, tail = src[:i], src[i:]
for a, b in rep:
    assert head.count(a) == 1, a
    head = head.replace(a, b)
src = head + tail
src += r'''
void csl_dbg(torch::Tensor d) { cudaMemcpyFromSymbol(d.data_ptr(), csl::g_dbg, sizeof(unsigned long long) * 512 * 4 * 13); }
'''
m = load_inline(name="s93_cslstamp_" + hashlib.md5(src.encode()).hexdigest()[:8], cpp_sources="void csl_decode(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
                "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, double); void csl_dbg(torch::Tensor);",
                cuda_sources=src, functions=["csl_decode", "csl_dbg"],
                extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"], verbose=False)
load, c = [x for x in bench.SCORED if x[1]["layer"] == "sliding" and x[1]["batch"] == 16 and x[1]["q_len"] == 1][0]
data = bench.data_for(load, c)
cfg = data["config"]
out = torch.empty((16, 1, 1024), dtype=torch.bfloat16, device="cuda")
cs = submission._cos_sin(cfg["rope_theta"], 64, "cuda")
def run():
    m.csl_decode(data["qkv"], cs, data["k_cache"], data["v_cache"], data["page_table"], data["seq_lens"], data["cache_loc"],
                 data["k_scale"], data["v_scale"], out, 1, cfg["sm_scale"] * cfg["qk_scale_factor"] * submission.LOG2E)
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda"); sink = torch.zeros((), dtype=torch.int64, device="cuda")
run(); torch.cuda.synchronize()
res = []
with open("/tmp/okc-gpu.lock", "w") as lk:
    fcntl.flock(lk, fcntl.LOCK_EX)
    for r in range(int(sys.argv[1]) if len(sys.argv) > 1 else 40):
        if not os.environ.get("WARM"): torch.sum(flush.view(torch.int64), 0, out=sink)
        run(); torch.cuda.synchronize()
        d = torch.zeros(512 * 4 * 13, dtype=torch.int64); m.csl_dbg(d)
        a = d.view(512, 4, 13).numpy().astype(np.float64)
        res.append((a - a[:, :, 0][a[:, :, 0] > 0].min()) / 1000.0)
a = np.mean(np.stack(res), 0)
names = ["start", "pagetable", "K landed+QK", "PV done", "premerge sync", "cluster wait", "pushed", "owner mbar ok", "out stored", "merge loop done", "shfl done", "pre-store", "before cwait"]
for k, nm in enumerate(names):
    v = a[(np.arange(512) % 16) < 8, 0, k] if 7 <= k < 12 else a[:, :, k].ravel()
    print(f"{nm:14s} min {v.min():5.2f} med {np.median(v):5.2f} p90 {np.percentile(v,90):5.2f} max {v.max():5.2f}")

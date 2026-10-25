"""agent2 s6: %globaltimer stamps in nvs's fused merge (verify 32): consumer start, loop done, spin done, merge done."""
import sys, torch, statistics, importlib
sys.path.insert(0, "/home/daytona/repo2/tools"); sys.path.insert(0, "/tmp/a2v")
import bench
src = open("/home/daytona/repo2/submission.py").read()
def rep(a, b):
    global src
    assert src.count(a) == 1, a; src = src.replace(a, b)
T = 'asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(_t));'
ST = lambda k: f'{{ unsigned long long _t; {T} if (lane == 0) reinterpret_cast<unsigned long long*>(cnt + 8192)[(blockIdx.x * 2 + h) * 8 + w * 0 + {k}] = _t; }}'
rep("    const int tid = threadIdx.x & 127, lane = tid & 31, w = tid >> 5;\n    const uint32_t qa",
    "    const int tid = threadIdx.x & 127, lane = tid & 31, w = tid >> 5;\n    if (w == split) " + ST(0) + "\n    const uint32_t qa")
rep("    const long long part = (((long long)b * 2 + h) * S + split) * MR;\n    const int ra",
    "    if (w == split) " + ST(1) + "\n    const long long part = (((long long)b * 2 + h) * S + split) * MR;\n    const int ra")
rep("    __syncwarp();\n    const long long part0", "    __syncwarp();\n    " + ST(2) + "\n    const long long part0")
rep("""                acc[2 * j] * rs / (1.f + __expf(-g.x)), acc[2 * j + 1] * rs / (1.f + __expf(-g.y)));
        }
    }
}""", """                acc[2 * j] * rs / (1.f + __expf(-g.x)), acc[2 * j + 1] * rs / (1.f + __expf(-g.y)));
        }
    }
    """ + ST(3) + "\n}")
rep("""        const float wo = ex2(mo - M);""", """        if (e == own0 ? 0 : 1) """ + ST(4).replace("{ unsigned long long _t;", "{ unsigned long long _t; asm volatile(\"\" :: \"f\"(M));") + """
        const float wo = ex2(mo - M);""")
rep("""        const float sc = vsv / den;""", """        """ + ST(5).replace("{ unsigned long long _t;", "{ unsigned long long _t; asm volatile(\"\" :: \"f\"(acc[15]), \"f\"(den));") + """
        const float sc = vsv / den;""")
rep("""            const float2 g = __bfloat1622float2(gt[e][j]);""", """            if (j == 0) """ + ST(6).replace("{ unsigned long long _t;", "{ unsigned long long _t; asm volatile(\"\" :: \"r\"(*(uint32_t*)&gt[e][7]), \"f\"(rs));") + """
            const float2 g = __bfloat1622float2(gt[e][j]);""")
rep('torch.zeros(8192, dtype=torch.int32', 'torch.zeros(8192 + 4096 * 4, dtype=torch.int32')
rep('name="muse_nvs_a2s6"', 'name="muse_nvs_a2s6st"')
open("/tmp/a2v/dmst.py", "w").write(src)
m = importlib.import_module("dmst")
for l, c in bench.SCORED:
    if l != "verify 32" or c["layer"] != "sliding": continue
    data = bench.data_for(l, c)
    fl = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    for _ in range(3): m.kernel_fn(data)
    for rep_ in range(3):
        fl.zero_(); m.kernel_fn(data); torch.cuda.synchronize()
        cnt = m._NVS_CNT[torch.device("cuda")] if torch.device("cuda") in m._NVS_CNT else list(m._NVS_CNT.values())[0]
        t = cnt[8192:].view(torch.int64)[:128 * 2 * 8].view(128 * 2, 8)[:, :7].double()
        z = t[:, 0].min(); t = (t - z) / 1000
        print(" | ".join(f"{nm} med {t[:, i].median():.2f} max {t[:, i].max():.2f}" for i, nm in enumerate(["start", "loop", "spun", "merged", "ml", "acc", "pre-store"])))

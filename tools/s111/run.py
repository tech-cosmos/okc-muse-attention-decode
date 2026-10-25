"""s111 native verify-full (nvf.cu): correctness through submission.kernel_fn with _gw_verify(BAL, LZ 10) swapped for
the native kernel, then paired cold-L2 timing of verify 16/32 full (Gluon vs native).
python tools/s111/run.py [test|time|both] [build args: ns pregs cregs lag pfd pk pipe]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch
import build, bench, reference, submission as S
mode = sys.argv[1] if len(sys.argv) > 1 else "both"
NV = build.build(*[int(a) for a in sys.argv[2:]], **({"ver": int(os.environ["VER"])} if "VER" in os.environ else {}))
_gw = S._gw_verify
USE = [True]


class _Swap:
    def __getitem__(self, grid):
        k = _gw[grid]
        def call(*a, **kw):
            if USE[0] and kw.get("BAL") and kw.get("LZ") == 10 and kw["HKV"] == 2 and kw["G"] == 8 and kw["L"] == 8 and kw["D"] == 64:
                qkv, kc, vc, ks, vs, pt, seq, loc, pacc, pml, qk = a
                NV.nvf(qkv, kc, vc, ks, pt, seq, pacc, pml, qk, kw["NP"], kw["NID"], kw["PAGE"])
                return
            return k(*a, **kw)
        return call


S._gw_verify = _Swap()
CASES = [dict(batch=32, ctx=80000, heads="tp8", layer="full", page=1, q_len=8, dist="coherent", spread="agentic"),
         dict(batch=16, ctx=80000, heads="tp8", layer="full", page=1, q_len=8, dist="coherent", spread="agentic"),
         dict(batch=16, ctx=9000, heads="tp8", layer="full", q_len=8, dist="edges", spread="mixed"),
         dict(batch=24, ctx=5000, heads="tp8", layer="full", q_len=8, dist="soft", kv_scale=0.015625, page=1, spread="mixed"),
         dict(batch=17, ctx=3000, heads="tp8", layer="full", q_len=8, dist="coherent", spread="mixed"),
         dict(batch=40, ctx=20000, heads="tp8", layer="full", q_len=8, dist="soft", page=1, spread="agentic"),
         dict(batch=16, ctx=80000, heads="tp8", layer="full", page=64, q_len=8, dist="edges", spread="agentic")]
if mode in ("test", "both"):
    for i, c in enumerate(CASES + [t for t in bench.TESTS]):
        data = reference.generate_input(seed=77 + i, **c)
        out = S.kernel_fn(data)
        torch.cuda.synchronize()
        ok, msg = reference.check_implementation(data, reference.collect(data, out))
        print("PASS" if ok else "FAIL", c, msg[:120], flush=True)
if mode in ("time", "both"):
    for load, c in bench.SCORED:
        if c["layer"] != "full" or c["q_len"] != 8 or c["batch"] < int(os.environ.get("MINB", "16")):
            continue
        for l2 in ([False, True] if os.environ.get("L2", "1") == "1" else [False]):
            d = dict(bench.data_for(load, c))
            if l2:
                d["page_table"] = d["page_table"] % 4096
            res = {0: [], 1: []}
            for rep in range(int(os.environ.get("REPS", "3"))):
                for u in ((0, 1) if rep % 2 == 0 else (1, 0)):
                    USE[0] = bool(u)
                    res[u].append(bench.time_graph(lambda: S.kernel_fn(d)) - bench.empty_cost())
            df = sorted(b - a for a, b in zip(res[0], res[1]))
            print(f"{load} full {'L2  ' if l2 else 'real'}: gluon {min(res[0]):.1f} native {min(res[1]):.1f} paired {df[len(df) // 2]:+.1f}", flush=True)

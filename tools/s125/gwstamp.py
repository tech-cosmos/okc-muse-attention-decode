"""s125: %globaltimer stamps per program inside _gw_verify (gbal) at verify full (H100, cold L2, graph replay).
Patches a copy of submission.py. Per program: [start, consumer-0 first tile ready, consumer-0 done, ws end, smid,
tiles, nseg]. usage: gwstamp.py batch ctx [reps]"""
import os, sys, statistics, importlib.util, fcntl
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import torch
src = open("/home/daytona/repo/submission.py").read()
TM = 'gl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=gl.int64, is_pure=False, pack=1)'
def rep(a, b):
    global src
    assert src.count(a) == 1, a
    src = src.replace(a, b)
# kernel start
a = """        h = gl.program_id(2) * 2
        # Let the merge kernel launch now (programmatic dependent launch); it waits for this grid."""
rep(a, f"        st_t0 = {TM}\n" + a)
# consumer signature + call
rep("""                         D: gl.constexpr, BN: gl.constexpr, NS: gl.constexpr, LZ: gl.constexpr = 0):
        \"\"\"_gw_consumer_cv (PEEL, LSUM)""", """                         D: gl.constexpr, BN: gl.constexpr, NS: gl.constexpr, LZ: gl.constexpr = 0, st_ptr=None):
        \"\"\"_gw_consumer_cv (PEEL, LSUM)""")
rep("""pml_ptr, 0, HQ, HKV, G, L, B, B_PAD, NP, NID, M, D, BN, NS, LZ)),""",
    """pml_ptr, 0, HQ, HKV, G, L, B, B_PAD, NP, NID, M, D, BN, NS, LZ, cnt_ptr)),""")
# first tile ready (LZ 10 first loop)
a = """                for i in range(gl.minimum(ntiles, LZ_PK)):
                    s = (it + i) % NS
                    mbarrier.wait(ready.index(s), ((it + i) // NS) & 1)
"""
rep(a, a + f"""                    if HEAD == 0:
                        if it + i == 0:
                            gl.store(st_ptr + pid * 8 + 1, {TM})
""")
a = """            it += ntiles
            l_i = gl.convert_layout(gl.sum(l8, 1), gl.SliceLayout(1, S_L))"""
rep(a, a)
a = """            _gw_store(pacc_ptr, pml_ptr, 0, h + HEAD, pid + b, m_i, l_i, acc, HKV, NID, M, D, gl.SliceLayout(1, S_L),
                      O_L)
"""
rep(a, a + f"""        if HEAD == 0:
            gl.store(st_ptr + pid * 8 + 2, {TM})
            gl.store(st_ptr + pid * 8 + 5, it.to(gl.int64))
            gl.store(st_ptr + pid * 8 + 6, nseg.to(gl.int64))
""")
a = """            ], [4, PW], [CREGS, PREGS])
        elif LAG:"""
rep(a, f"""            ], [4, PW], [CREGS, PREGS])
            gl.store(cnt_ptr + split * 8 + 3, {TM})
            gl.store(cnt_ptr + split * 8 + 0, st_t0)
            gl.store(cnt_ptr + split * 8 + 4, gl.inline_asm_elementwise("mov.u32 $0, %smid;", "=r", [], dtype=gl.int32, is_pure=False, pack=1).to(gl.int64))
        elif LAG:""")
rep("""        LAG=plan.get("lag", 1), BAL=True, B=batch,""", """        LAG=plan.get("lag", 1), BAL=True, cnt_ptr=_STAMP, B=batch,""")
src = src.replace("import torch\n", "import torch\n_STAMP = torch.zeros(4096 * 8, dtype=torch.int64, device='cuda')\n", 1)
path = "/tmp/s125_gwstamp_sub.py"; open(path, "w").write(src)
spec = importlib.util.spec_from_file_location("submission", path); sub = importlib.util.module_from_spec(spec)
sys.modules["submission"] = sub; spec.loader.exec_module(sub)
import bench
batch, ctx = int(sys.argv[1]), int(sys.argv[2]); reps = int(sys.argv[3]) if len(sys.argv) > 3 else 10
c = dict(batch=batch, ctx=ctx, heads="tp8", layer="full", page=1, q_len=8, dist="coherent", spread="agentic")
d = bench.data_for(f"b{batch}", c)
fn = lambda: sub.kernel_fn(d)
fn(); torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g): fn()
fl = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
rows = []
lk = open("/tmp/okc-gpu.lock", "w"); fcntl.flock(lk, fcntl.LOCK_EX)
for _ in range(reps):
    fl.add_(1); torch.cuda.synchronize()
    sub._STAMP.zero_(); torch.cuda.synchronize()
    g.replay(); torch.cuda.synchronize()
    rows.append(sub._STAMP.view(-1, 8).cpu().clone())
fcntl.flock(lk, fcntl.LOCK_UN)
import numpy as np
n = int((rows[0][:, 0] > 0).sum())
def q(x): return " ".join(f"{np.percentile(x, p):7.2f}" for p in (0, 10, 50, 90, 100))
acc = {k: [] for k in ("start", "first", "cdone", "end")}
for st in rows:
    st = st[:n].numpy().astype(np.float64); t0 = st[:, 0].min()
    for k, i in (("start", 0), ("first", 1), ("cdone", 2), ("end", 3)): acc[k].append((st[:, i] - t0) / 1e3)
print(f"B{batch} ctx{ctx}: {n} programs, tiles/prog {np.median(rows[0][:n,5].numpy())}, nseg max {rows[0][:n,6].max().item()}")
print("phase     min    p10    p50    p90    max  (us from first start, pooled over reps)")
for k, v in acc.items(): print(f"{k:6s} {q(np.concatenate(v))}")
st = rows[-1][:n].numpy().astype(np.float64); t0 = st[:, 0].min()
sm = st[:, 4]; cd = (st[:, 3] - t0) / 1e3
per_sm = {}
for s_, e_ in zip(sm, cd): per_sm.setdefault(int(s_), []).append(e_)
print("progs/SM counts:", np.bincount([len(v) for v in per_sm.values()]))
# correlations over all reps
S0, F0, E0, SMs = [], [], [], []
for st in rows:
    st = st[:n].numpy().astype(np.float64); t0 = st[:, 0].min()
    S0.append((st[:, 0] - t0) / 1e3); F0.append((st[:, 1] - t0) / 1e3); E0.append((st[:, 3] - t0) / 1e3); SMs.append(st[:, 4])
S0, F0, E0 = map(np.concatenate, (S0, F0, E0)); SMs = np.concatenate(SMs)
print("corr(end,start) %.2f corr(end,first) %.2f" % (np.corrcoef(E0, S0)[0, 1], np.corrcoef(E0, F0)[0, 1]))
late = S0 > 1.0
print("late starters %d/%d: end mean %.2f vs others %.2f" % (late.sum(), len(S0), E0[late].mean(), E0[~late].mean()))
print("end - start: p10 %.2f p50 %.2f p90 %.2f max %.2f" % tuple(np.percentile(E0 - S0, [10, 50, 90, 100])))
# per-program-id consistency: is the same program slow every rep?
E = np.stack([((st[:n].numpy()[:, 3] - st[:n].numpy()[:, 0].min()) / 1e3) for st in rows])
pm = E.mean(0); print("per-program mean end: p10 %.2f p50 %.2f max %.2f; rep-to-rep std median %.2f" % (np.percentile(pm, 10), np.median(pm), pm.max(), np.median(E.std(0))))
print("slowest 10 program ids:", np.argsort(pm)[-10:], "their tiles", rows[0][np.argsort(pm)[-10:], 5].numpy())

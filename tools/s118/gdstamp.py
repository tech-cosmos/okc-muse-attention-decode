"""s118: %globaltimer stamps per program inside _gd_full at decode 1 full (H100, cold L2, graph replay).
Patches a copy of submission.py: stamps [start, first tile landed, loop done, end, smid] per program into a buffer
passed as cs_ptr (unused on full layers: no RoPE). usage: gdstamp.py [reps]; env WARM=1 = no flush."""
import os, sys, re, statistics, importlib.util, fcntl
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import torch
src = open("/home/daytona/repo/submission.py").read()
TM = 'gl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l", [], dtype=gl.int64, is_pure=False, pack=1)'
a = """        gl.inline_asm_elementwise("griddepcontrol.launch_dependents;", "=r", [], dtype=gl.int32, is_pure=False,
                                  pack=1)
        C: gl.constexpr = 2 * D
        M: gl.constexpr = 2 * G"""
assert src.count(a) == 1
src = src.replace(a, a + f"\n        st_t0 = {TM}\n        st_t1 = st_t0")
b = """            cp.wait_group(NS - 1)
            gl.barrier()
            s = i % NS
            kb = k_ring.index(s).permute((1, 0)).load(KB)"""
assert src.count(b) == 1
src = src.replace(b, b.replace("            s = i % NS", f"            st_now = {TM}\n            st_t1 = gl.where(i == 0, st_now, st_t1)\n            s = i % NS"))
c = """        cp.wait_group(0)

        # Partials [B, HKV, S + 1, G, D]"""
assert src.count(c) == 1
src = src.replace(c, c.replace("cp.wait_group(0)", f"cp.wait_group(0)\n        st_t2 = {TM}"))
d = """        if RED:
            if epi:
                _gd_red_add(acc, m_i, l_i, flg, split, kid, red_ptr, rcnt_ptr, M, D, NR)
"""
assert src.count(d) == 1
src = src.replace(d, d + f"""        st_t3 = {TM}
        st_sm = gl.inline_asm_elementwise("mov.u32 $0, %smid;", "=r", [], dtype=gl.int32, is_pure=False, pack=1)
        if epi:
            sp = cs_ptr + gl.program_id(0) * 8
            gl.store(sp, st_t0)
            gl.store(sp + 1, st_t1)
            gl.store(sp + 2, st_t2)
            gl.store(sp + 3, st_t3)
            gl.store(sp + 4, st_sm.to(gl.int64))
""")
e = '_cos_sin(cfg["rope_theta"], d, device) if cfg["rope_theta"] else qkv,'
src = src.replace(e, '_cos_sin(cfg["rope_theta"], d, device) if cfg["rope_theta"] else _STAMP,')
TT = TM.replace("gl.", "tl.")
f0 = "             SL: tl.constexpr = 0, RCW: tl.constexpr = 0, BAL: tl.constexpr = False):"
assert src.count(f0) == 1
src = src.replace(f0, f0[:-2] + ", st_ptr=None):")
f1 = """    vs = tl.load(vs_ptr)
    gdc_wait()
"""
assert src.count(f1) == 1
src = src.replace(f1, f"    st_a = {TT}\n" + f1 + f"    st_b = {TT}\n")
f4 = """    nf = tl.load(rcnt_ptr + 1)
    tl.debug_barrier()  # every"""
assert src.count(f4) == 1
src = src.replace(f4, f4.replace("    tl.debug_barrier()  # every", f"    tl.debug_barrier()\n    st_d = {TT}\n    tl.debug_barrier()  # every"))
f5 = "    o = a / l[:, None] * vs\n"
assert src.count(f5) == 1
src = src.replace(f5, f"    tl.debug_barrier()\n    st_e = {TT}\n" + f5)
f2 = "    tl.store(out_ptr + col, (o * tl.sigmoid(gate)).to(tl.bfloat16))\n\n\ndef _verify_nvs"
assert src.count(f2) == 1
src = src.replace(f2, f2.replace("\n\n\ndef", f"\n    tl.debug_barrier()\n    st_c = {TT}\n    tl.store(st_ptr + 8000, st_a)\n    tl.store(st_ptr + 8001, st_b)\n    tl.store(st_ptr + 8002, st_c)\n    tl.store(st_ptr + 8003, st_d)\n    tl.store(st_ptr + 8004, st_e)\n\n\ndef"))
f3 = "RF_NW, launch_pdl=True)"  # the _red_fin launch
assert src.count(f3) == 1
src = src.replace(f3, "RF_NW, launch_pdl=True, st_ptr=_STAMP)")
src += "\n_STAMP = None\n"
open("/tmp/sub_stamp.py", "w").write(src)
spec = importlib.util.spec_from_file_location("submission", "/tmp/sub_stamp.py")
sub = importlib.util.module_from_spec(spec); sys.modules["submission"] = sub; spec.loader.exec_module(sub)
import bench
load, cfgd = [x for x in bench.SCORED if f"{x[0]} {x[1]['layer']}" == "decode 1 full"][0]
data = bench.data_for(load, cfgd)
sub._STAMP = torch.zeros(8192, dtype=torch.int64, device="cuda")
fn = lambda: sub.kernel_fn(data)
fn(); torch.cuda.synchronize()
g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    fn()
    with torch.cuda.graph(g, stream=s):
        fn()
torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda"); sink = torch.zeros((), dtype=torch.int64, device="cuda")
reps = int(sys.argv[1]) if len(sys.argv) > 1 else 20
rows = []
with open("/tmp/okc-gpu.lock", "w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    for _ in range(reps):
        if not os.environ.get("WARM"):
            torch.sum(flush.view(torch.int64), 0, out=sink)
        sub._STAMP.zero_()
        ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ev0.record(); g.replay(); ev1.record(); torch.cuda.synchronize()
        st = sub._STAMP.view(-1, 8)[:1000, :5].cpu()
        st = st[st[:, 0] > 0].double(); n = st.shape[0]
        t0 = st[:, 0].min()
        rel = (st[:, :4] - t0) / 1000.0
        fin = (sub._STAMP[8000:8005].cpu().double() - t0) / 1000.0
        rows.append((ev0.elapsed_time(ev1) * 1000, n, rel, st[:, 4], fin))
def q(x, p): return float(torch.quantile(x, p))
for name, k in [("start", 0), ("first tile", 1), ("loop done", 2), ("end", 3)]:
    xs = torch.stack([r[2][:, k] for r in rows])
    print(f"{name:11s} min {xs.min(1).values.mean():6.2f} p10 {torch.tensor([q(r[2][:, k], .1) for r in rows]).mean():6.2f} "
          f"p50 {torch.tensor([q(r[2][:, k], .5) for r in rows]).mean():6.2f} p90 {torch.tensor([q(r[2][:, k], .9) for r in rows]).mean():6.2f} "
          f"max {xs.max(1).values.mean():6.2f}")
dur = torch.stack([r[2][:, 3] - r[2][:, 0] for r in rows])
print("per-program duration: p10 %.2f p50 %.2f p90 %.2f max %.2f" % (dur.quantile(.1), dur.quantile(.5), dur.quantile(.9), dur.max(1).values.mean()))
fl = torch.stack([r[2][:, 1] - r[2][:, 0] for r in rows])
print("start -> first tile: p50 %.2f p90 %.2f" % (fl.quantile(.5), fl.quantile(.9)))
print("graph event time median %.2f us, programs %d" % (statistics.median(r[0] for r in rows), rows[0][1]))
fin = torch.stack([r[4] for r in rows]).mean(0)
print("_red_fin: start %.2f  after gdc_wait %.2f  end %.2f  replicas loaded %.2f  zeroed+list %.2f (us from first _gd_full program start)" % tuple(fin.tolist()))
# start-time histogram (0.25 us bins) of the last rep, and programs per SM
r = rows[-1]
h = torch.histc(r[2][:, 0], bins=24, min=0, max=6); print("start hist (0.25us bins):", h.int().tolist())
h = torch.histc(r[2][:, 3], bins=48, min=0, max=12); print("end hist (0.25us bins):", h.int().tolist())
cnt = torch.bincount(r[3].long()); print("progs per SM: min %d max %d, SMs %d" % (cnt[cnt > 0].min(), cnt.max(), (cnt > 0).sum()))
# end time vs programs on the same SM (all reps)
by = {}
for rr in rows:
    sm = rr[3].long(); cnt = torch.bincount(sm)
    keyed = (rr[2][:, 2] - rr[2][:, 1]) > 0.5  # programs with key tiles
    for k in range(sm.shape[0]):
        if keyed[k]:
            by.setdefault(int(cnt[sm[k]]), []).append((float(rr[2][k, 3]), float(rr[2][k, 1])))
for n, v in sorted(by.items()):
    e = torch.tensor(v)
    print(f"SM with {n} progs: n {len(v)//len(rows)}/rep, end p50 {e[:, 0].median():.2f} p90 {e[:, 0].quantile(.9):.2f}, first tile p50 {e[:, 1].median():.2f}")

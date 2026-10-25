"""agent2: per-program %globaltimer start/end + %smid of _attn_split on one scored shape (cold L2, graph replay).
  python tools/a2/stamp.py 'verify 32' sliding '{}'"""
import sys, json, importlib, torch
sys.path.insert(0, "/home/daytona/repo2/tools"); sys.path.insert(0, "/home/daytona/repo2")
src = open("/home/daytona/repo2/submission.py").read()
A = "    if SPEC > 0:\n        real = tl.load(seq_ptr + tl.program_id(1))"
assert src.count(A) == 1
GT = 'tl.inline_asm_elementwise("mov.u64 $0, %globaltimer;", "=l,r", [tl.program_id(0)], dtype=tl.int64, is_pure=False, pack=1)'
SM = 'tl.inline_asm_elementwise("mov.u32 $0, %smid;", "=r,r", [tl.program_id(0)], dtype=tl.int32, is_pure=False, pack=1)'
src = src.replace(A, f"    t0 = {GT}\n" + A)
B = "        _attn_body(0, False, qkv_ptr, cs_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, pt_ptr, seq_ptr, loc_ptr, pacc_ptr, pml_ptr, cnt_ptr, out_ptr, qk_scale, HQ, HKV, HPP, G, L, MG, M_PAD, L_PAD, D, PAGE, MAX_PAGES, WINDOW, ROPE, S, CHUNK, BLOCK_N, STAGES, FUSE, S_PAD, QK8, PDL, DOT, PEEL, DUAL, BALANCE, SEP, PF, PRE, MIF, PREQ)\n"
assert src.count(B) == 1
src = src.replace(B, B + f"""    t1 = {GT}
    sid = (tl.program_id(2) * tl.num_programs(1) + tl.program_id(1)) * tl.num_programs(0) + tl.program_id(0)
    sp = (cnt_ptr + 8192).to(tl.pointer_type(tl.int64))
    tl.store(sp + sid * 4, t0)
    tl.store(sp + sid * 4 + 1, t1)
    tl.store(sp + sid * 4 + 2, {SM}.to(tl.int64))
""")
import os
for a_, b_ in json.loads(os.environ.get("A2REP", "[]")):
    assert src.count(a_) >= 1, a_
    src = src.replace(a_, b_)
open("/tmp/a2stamp_sub.py", "w").write(src)
sys.path.insert(0, "/tmp")
import bench
sub = importlib.import_module("a2stamp_sub")
bench.submission = sub
buf = torch.zeros(8192 + 4 * 8192 * 2, dtype=torch.int32, device="cuda")
sub._counters = lambda n, device, tag="": buf
load, layer, plan = sys.argv[1], sys.argv[2], json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
for l, c in bench.SCORED:
    if l == load and c["layer"] == layer:
        data = bench.data_for(l, c)
        print("graph time", round(bench.run_shape(l, c, plan), 2))
        orig = sub._plan
        sub._plan = lambda cfg, mp, dev: {**orig(cfg, mp, dev), **plan}
        flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
        for _ in range(3):
            flush.zero_(); sub.kernel_fn(data)
        torch.cuda.synchronize()
st = buf[8192:].view(torch.int64).view(-1, 4).cpu()
st = st[st[:, 0] > 0]
t0 = st[:, 0].min()
s, e, sm = (st[:, 0] - t0).float() / 1e3, (st[:, 1] - t0).float() / 1e3, st[:, 2]
d = e - s
print(f"programs {len(st)}  SMs used {len(set(sm.tolist()))}")
print(f"start: min {s.min():.2f} med {s.median():.2f} max {s.max():.2f} us")
print(f"end:   min {e.min():.2f} med {e.median():.2f} p90 {e.quantile(.9):.2f} max {e.max():.2f}")
import collections
S_ = int(os.environ.get("A2S", "4"))
for k in range(S_): print(f"split {k} dur med {d[torch.arange(len(d)) % S_ == k].median():.2f} max {d[torch.arange(len(d)) % S_ == k].max():.2f}")
print(f"dur:   min {d.min():.2f} med {d.median():.2f} p90 {d.quantile(.9):.2f} max {d.max():.2f}")
torch.save(st, "/tmp/a2stamp.pt")

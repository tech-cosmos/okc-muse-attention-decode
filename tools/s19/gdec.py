"""Session 19 prototype: decode (q_len 1) full layers as a Gluon kernel with native FP8 mma.sync for QK^T
(Triton's MMAv2 path upcasts every K byte to FP16 on sm_90) and an explicit cp.async ring.

  python tools/s19/gdec.py test            # decode 16/32 full + extra cases vs reference
  python tools/s19/gdec.py time '<json>'   # decode 16/32 full, real and L2-resident, Triton vs Gluon
"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import torch, triton  # noqa: E402
import bench, reference, submission  # noqa: E402
from submission import gluon, gl, cp, _new_tokens, _combine, _combine_hb, DOT_DTYPE, LOG2E, USE_PDL  # noqa: E402
from triton.experimental.gluon.language.nvidia.ampere import mma_v2  # noqa: E402


@gluon.jit
def _gd_full(qkv_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, pt_ptr, seq_ptr, loc_ptr, pacc_ptr, pml_ptr, qk_scale,
             HQ: gl.constexpr, HKV: gl.constexpr, G: gl.constexpr, D: gl.constexpr, PAGE: gl.constexpr,
             MAX_PAGES: gl.constexpr, S: gl.constexpr, BN: gl.constexpr, NS: gl.constexpr, VCV: gl.constexpr = 0, PF: gl.constexpr = 1, NEWT: gl.constexpr = False, dbg_ptr=None):
    split = gl.program_id(0)
    b = gl.program_id(1)
    h = gl.program_id(2) * 2
    gl.inline_asm_elementwise("griddepcontrol.launch_dependents;", "=r", [], dtype=gl.int32, is_pure=False, pack=1)
    C: gl.constexpr = 2 * D
    M: gl.constexpr = 2 * G
    QKV_W: gl.constexpr = (2 * HQ + 2 * HKV) * D
    ROW_W: gl.constexpr = HKV * D
    MMA: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[1, 1], instr_shape=[16, 8])
    QA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=MMA, k_width=4)
    KB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=MMA, k_width=4)
    PA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=MMA, k_width=2)
    VB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=MMA, k_width=2)
    BL: gl.constexpr = gl.BlockedLayout([1, 16], [4, 8], [1, 1], [1, 0])
    QL: gl.constexpr = BL
    SH: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 8, [1, 0])

    seq = gl.load(seq_ptr + b)
    chunk = gl.cdiv(gl.cdiv(seq, S), BN) * BN
    lo = split * chunk
    hi = gl.minimum(lo + chunk, seq)
    ntiles = gl.cdiv(gl.maximum(hi - lo, 0), BN)
    pt_row = pt_ptr + b.to(gl.int64) * MAX_PAGES
    kbase = kc_ptr + h * D
    vbase = vc_ptr + h * D
    k_ring = gl.allocate_shared_memory(gl.float8e4nv, [NS, BN, C], SH)
    v_ring = gl.allocate_shared_memory(gl.float8e4nv, [NS, BN, C], SH)
    n_g = gl.arange(0, BN, layout=gl.SliceLayout(1, BL))
    c_g = gl.arange(0, C, layout=gl.SliceLayout(0, BL))
    # Issue the first NS - 1 tiles (rows past hi are not read: cp.async zero-fills them).
    for j in gl.static_range(NS - 1):
        t = lo + j * BN + n_g
        ok = t < hi
        pg = gl.load(pt_row + t // PAGE, mask=ok, other=0)
        off = ((pg.to(gl.int64) * PAGE + t % PAGE) * ROW_W)[:, None] + c_g[None, :]
        cp.async_copy_global_to_shared(k_ring.index(j), kbase + off, mask=ok[:, None] & (c_g[None, :] >= 0))
        cp.async_copy_global_to_shared(v_ring.index(j), vbase + off, mask=ok[:, None] & (c_g[None, :] >= 0))
        cp.commit_group()
    t = lo + (NS - 1) * BN + n_g
    pg = gl.load(pt_row + t // PAGE, mask=t < hi, other=0)
    pg2 = pg
    if PF == 2:
        pg2 = gl.load(pt_row + (t + BN) // PAGE, mask=t + BN < hi, other=0)

    # Block-diagonal queries [2G, 2D]: row r is query head (h + r // G) * G + r % G, nonzero only in its own
    # KV head's D columns of the 128-byte row. RMSNorm, then E4M3.
    r = gl.arange(0, M, layout=gl.SliceLayout(1, QL))
    cq = gl.arange(0, C, layout=gl.SliceLayout(0, QL))
    hp = r // G
    q_ok = (cq // D)[None, :] == hp[:, None]
    x = gl.load(qkv_ptr + b * QKV_W + ((h + hp) * G + r % G)[:, None] * D + (cq % D)[None, :], mask=q_ok,
                other=0.0).to(gl.float32)
    x = gl.where(q_ok, x, 0.0)
    x = x * gl.rsqrt(gl.sum(x * x, 1) / D + 1e-5)[:, None]
    q = gl.convert_layout(x.to(gl.float8e4nv), QA)
    scale = qk_scale * gl.load(ks_ptr)
    if dbg_ptr is not None:
        if (split == 0) & (b == 0):
            gl.store(dbg_ptr + 1024 + r[:, None] * C + cq[None, :], x.to(gl.float8e4nv).to(gl.float32))

    m_i = gl.full([M], -1.0e30, gl.float32, layout=gl.SliceLayout(1, MMA))
    l_i = gl.zeros([M], gl.float32, layout=gl.SliceLayout(1, MMA))
    acc = gl.zeros([M, C], gl.float32, layout=MMA)
    n_s = gl.arange(0, BN, layout=gl.SliceLayout(0, MMA))
    for i in range(ntiles):
        sj = (i + NS - 1) % NS
        ok = t < hi
        off = ((pg.to(gl.int64) * PAGE + t % PAGE) * ROW_W)[:, None] + c_g[None, :]
        gl.barrier()  # slot sj was read last iteration
        cp.async_copy_global_to_shared(k_ring.index(sj), kbase + off, mask=ok[:, None] & (c_g[None, :] >= 0))
        cp.async_copy_global_to_shared(v_ring.index(sj), vbase + off, mask=ok[:, None] & (c_g[None, :] >= 0))
        cp.commit_group()
        t = t + BN
        if PF == 2:
            pg = pg2
            pg2 = gl.load(pt_row + (t + BN) // PAGE, mask=t + BN < hi, other=0)
        else:
            pg = gl.load(pt_row + t // PAGE, mask=t < hi, other=0)
        cp.wait_group(NS - 1)
        gl.barrier()
        s = i % NS
        kt = k_ring.index(s).permute((1, 0)).load(KB)
        sc = mma_v2(q, kt, gl.zeros([M, BN], gl.float32, MMA))
        if dbg_ptr is not None:
            if (i == 0) & (split == 0) & (b == 0):
                ro_ = gl.arange(0, M, layout=gl.SliceLayout(1, MMA))
                gl.store(dbg_ptr + ro_[:, None] * BN + n_s[None, :], sc)
        if lo + (i + 1) * BN > hi:
            sc = gl.where((lo + i * BN + n_s < hi)[None, :], sc, float("-inf"))
        m_new = gl.maximum(m_i, gl.max(sc, 1) * scale)
        alpha = gl.exp2(m_i - m_new)
        p = gl.exp2(gl.fma(sc, scale, -m_new[:, None]))
        l_i = l_i * alpha + gl.sum(p, 1)
        m_i = m_new
        acc = acc * alpha[:, None]
        if VCV == 1:
            v = gl.convert_layout(v_ring.index(s).load(BL).to(gl.float16), VB)
        else:
            v = v_ring.index(s).load(VB).to(gl.float16)
        acc = mma_v2(gl.convert_layout(p.to(gl.float16), PA), v, acc)
    cp.wait_group(0)

    # Partials: row r's own head's D columns.
    ro = gl.arange(0, M, layout=gl.SliceLayout(1, MMA))
    co = gl.arange(0, C, layout=gl.SliceLayout(0, MMA))
    hh = h + ro // G
    part = ((b * HKV + hh) * (S + 1) + split) * G + ro % G
    st_ok = (co // D)[None, :] == (ro // G)[:, None]
    gl.store(pacc_ptr + part[:, None] * D + (co % D)[None, :], acc.to(pacc_ptr.dtype.element_ty), mask=st_ok)
    gl.store(pml_ptr + part * 2, m_i)
    gl.store(pml_ptr + part * 2 + 1, l_i)
    if NEWT:
        if split == S - 1:
            # The step's own token (both KV heads): normalize K, store K and V as FP8 at cache_loc, and write its
            # one-key partial (max = score, sum = 1, acc = V) into slot S. Rows hold their own head's columns.
            loc = gl.load(loc_ptr + b)
            kx = gl.load(qkv_ptr + b * QKV_W + (HQ + h) * D + cq)[None, :].to(gl.float32) + gl.zeros([M, C], gl.float32, QL)
            kx = gl.where(q_ok, kx, 0.0)
            k_scale = gl.load(ks_ptr)
            kq = (kx * gl.rsqrt(gl.sum(kx * kx, 1) / D + 1e-5)[:, None] / k_scale).to(gl.float8e4nv)
            vx = gl.load(qkv_ptr + b * QKV_W + (HQ + HKV + h) * D + cq)[None, :].to(gl.float32) + gl.zeros([M, C], gl.float32, QL)
            vq = (gl.where(q_ok, vx, 0.0) / gl.load(vs_ptr)).to(gl.float8e4nv)
            first = q_ok & ((r % G) == 0)[:, None]
            dst = (loc.to(gl.int64) * ROW_W + h * D + cq)[None, :] + gl.zeros([M, C], gl.int64, QL)
            gl.store(kc_ptr + dst, kq, mask=first)
            gl.store(vc_ptr + dst, vq, mask=first)
            sn = gl.sum(x.to(gl.float8e4nv).to(gl.float32) * kq.to(gl.float32), 1) * scale
            pn = ((b * HKV + h + hp) * (S + 1) + S) * G + r % G
            gl.store(pacc_ptr + pn[:, None] * D + (cq % D)[None, :], vq.to(gl.float32).to(pacc_ptr.dtype.element_ty),
                     mask=q_ok)
            gl.store(pml_ptr + pn * 2, sn)
            gl.store(pml_ptr + pn * 2 + 1, gl.full([M], 1.0, gl.float32, gl.SliceLayout(1, QL)))
    else:
        # _new_tokens (launched just before, PDL) must be done before the merge reads slot S.
        gl.inline_asm_elementwise("griddepcontrol.wait;", "=r", [], dtype=gl.int32, is_pure=False, pack=1)


PLAN = dict(splits=32, block_n=32, stages=3, pdt="f32", merge="combine")


def decode_full(data, plan=None):
    plan = {**PLAN, **(plan or {})}
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    group = hq // hkv
    rows = length * group
    qkv = data["qkv"]
    device = qkv.device
    page_table = data["page_table"]
    splits = plan["splits"]
    pacc = torch.empty((batch, hkv, splits + 1, rows, d), dtype=submission._PDT[plan["pdt"]], device=device)
    pml = torch.empty((batch, hkv, splits + 1, rows, 2), dtype=torch.float32, device=device)
    out = torch.empty((batch, length, hq * d), dtype=torch.bfloat16, device=device)
    qk_scale = cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E
    if not plan.get("only") and not plan.get("newt"):
      _new_tokens[(batch, hkv)](
        qkv, data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"], data["seq_lens"], data["cache_loc"],
        pacc, pml, qk_scale, HQ=hq, HKV=hkv, G=group, L=length, MG=rows, L_PAD=16, D=d, S=splits, DOT=DOT_DTYPE,
        num_warps=4)
    _gd_full[(splits, batch, hkv // 2)](
        qkv, data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"], page_table, data["seq_lens"],
        data["cache_loc"], pacc, pml, qk_scale,
        HQ=hq, HKV=hkv, G=group, D=d, PAGE=cfg["page_size"], MAX_PAGES=page_table.shape[1], S=splits,
        BN=plan["block_n"], NS=plan["stages"], VCV=plan.get("vcv", 0), PF=plan.get("pf", 1), NEWT=plan.get("newt", False), dbg_ptr=plan.get("dbg"), num_warps=1, launch_pdl=not plan.get("newt"), **({"maxnreg": plan["maxnreg"]} if plan.get("maxnreg") else {}))
    s = splits + 1
    if plan.get("only"):
        return out
    _combine[(batch * length, hq)](
        qkv, pacc, pml, data["v_scale"], out,
        HQ=hq, HKV=hkv, G=group, L=length, M=rows, D=d, S=s, S_PAD=triton.next_power_of_2(s),
        SB=min(128, triton.next_power_of_2(s)), PDL=USE_PDL, num_warps=4 if s > 32 else 2, launch_pdl=USE_PDL)
    return out


CASES = [dict(batch=16, ctx=80000, heads="tp8", layer="full", page=1, spread="agentic", dist="coherent"),
         dict(batch=32, ctx=80000, heads="tp8", layer="full", page=1, spread="agentic", dist="coherent"),
         dict(batch=8, ctx=8192, heads="tp8", layer="full", spread="mixed"),
         dict(batch=4, ctx=3000, heads="tp8", layer="full"),
         dict(batch=2, ctx=9000, heads="tp8", layer="full", kv_scale=0.015625),
         dict(batch=4, ctx=80000, heads="tp8", layer="full", spread="mixed", dist="soft"),
         dict(batch=4, ctx=80000, heads="tp8", layer="full", spread="mixed", dist="coherent", page=1),
         dict(batch=16, ctx=5000, heads="tp8", layer="full", dist="edges", page=1)]


def test(plan=None):
    bad = 0
    for i, c in enumerate(CASES):
        data = reference.generate_input(seed=777 + i, **c)
        out = decode_full(data, plan)
        torch.cuda.synchronize()
        ok, msg = reference.check_implementation(data, reference.collect(data, out))
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'} {c} {msg[:100]}", flush=True)
    return bad


def timing(plan=None):
    for load, c in bench.SCORED:
        if c["layer"] != "full" or c["q_len"] != 1 or c["batch"] == 1:
            continue
        data = bench.data_for(load, c)
        d2 = dict(data)
        d2["page_table"] = data["page_table"] % 4096
        e = bench.empty_cost()
        tt = bench.time_graph(lambda: submission.kernel_fn(data)) - e
        tt2 = bench.time_graph(lambda: submission.kernel_fn(d2)) - e
        tg = bench.time_graph(lambda: decode_full(data, plan)) - e
        tg2 = bench.time_graph(lambda: decode_full(d2, plan)) - e
        print(f"{load} full: triton {tt:7.2f} (L2 {tt2:7.2f})  gluon {tg:7.2f} (L2 {tg2:7.2f})", flush=True)


if __name__ == "__main__":
    plan = json.loads(sys.argv[2]) if len(sys.argv) > 2 else None
    if sys.argv[1] == "test":
        sys.exit(test(plan))
    if sys.argv[1] == "time":
        timing(plan)


def emu_timing(plan=None, block=40):
    """Triton vs Gluon under tools/emu.py (BLOCK SMs parked); splits scaled by (132 - block) / 132 for both."""
    import emu
    f = (132 - block) / 132
    orig = submission._plan
    submission._plan = lambda cfg, mp, dev: {**orig(cfg, mp, dev), "splits": round(orig(cfg, mp, dev)["splits"] * f)}
    x = torch.zeros(1, device="cuda")
    submission._SMS[0] = 132 - block
    empty = emu.time_blocked(lambda: x.add_(1), block, reps=60)
    for load, c in bench.SCORED:
        if c["layer"] != "full" or c["q_len"] != 1 or c["batch"] == 1:
            continue
        data = bench.data_for(load, c)
        tt = emu.time_blocked(lambda: submission.kernel_fn(data), block, 20) - empty
        p = {**PLAN, **(plan or {})}
        p["splits"] = round((64 if c["batch"] == 16 else 32) * f)
        tg = emu.time_blocked(lambda: decode_full(data, p), block, 20) - empty
        print(f"emu {132 - block} SMs {load} full: triton {tt:7.2f}  gluon {tg:7.2f}", flush=True)


if __name__ == "__main__" and sys.argv[1] == "emu":
    emu_timing(plan)

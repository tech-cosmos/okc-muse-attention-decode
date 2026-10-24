"""Prototype: Gluon verify-full (cached keys only) with a warp-specialized gather producer.

Default partition (4 warps): queries in smem, per tile wait `ready`, two QK^T wgmmas (per-head K slices),
V fp8 -> fp16 staging, release the ring slot (`empty`), online softmax, P@V.
Worker partition (PW warps): page indices, cp.async of the 128-byte K and V rows into the ring, and
cp.async.mbarrier.arrive.noinc on `ready`. Writes the same partials as gluon_verify._gv2.

  python tools/gluon_ws.py "verify 16" S NS MAXNREG PW PREGS [CREGS: >0 = _gws2, one consumer warpgroup per head]
"""
import sys
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (fence_async_shared, warpgroup_mma, warpgroup_mma_wait,
                                                                mbarrier)
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp

sys.path.insert(0, ".")
sys.path.insert(0, "tools")
from gluon_verify import _softmax, _store_part, triton_partials, merged, time_fn  # noqa: E402


@gluon.jit
def _producer(pt_row, kbase, vbase, k_ring, v_ring, ready, empty, lo, hi, ntiles,
              ROW_W: gl.constexpr, BN: gl.constexpr, C: gl.constexpr, NS: gl.constexpr, PW: gl.constexpr):
    GLY: gl.constexpr = gl.BlockedLayout([1, 16], [4, 8], [PW, 1], [1, 0])
    n_g = gl.arange(0, BN, layout=gl.SliceLayout(1, GLY))
    c_g = gl.arange(0, C, layout=gl.SliceLayout(0, GLY))
    pg = gl.load(pt_row + gl.minimum(lo + n_g, hi - 1))
    for i in range(ntiles):
        s = i % NS
        off = (pg.to(gl.int64) * ROW_W)[:, None] + c_g[None, :]
        pg = gl.load(pt_row + gl.minimum(lo + (i + 1) * BN + n_g, hi - 1))
        mbarrier.wait(empty.index(s), ((i // NS) & 1) ^ 1)
        cp.async_copy_global_to_shared(k_ring.index(s), kbase + off)
        cp.async_copy_global_to_shared(v_ring.index(s), vbase + off)
        cp.mbarrier_arrive(ready.index(s), increment_count=False)


@gluon.jit
def _consumer(q0, q1, k_ring, v_ring, vh0, vh1, ready, empty, lo, hi, ntiles, scale,
              M: gl.constexpr, D: gl.constexpr, BN: gl.constexpr, NS: gl.constexpr):
    VL: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    S_L: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, BN, 32])
    O_L: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, D, 16])
    P_L: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=O_L, k_width=2)
    m0 = gl.full([M], -1.0e30, gl.float32, layout=gl.SliceLayout(1, S_L))
    l0 = gl.zeros([M], gl.float32, layout=gl.SliceLayout(1, S_L))
    a0 = gl.zeros([M, D], gl.float32, layout=O_L)
    m1, l1, a1 = m0, l0, a0
    n_s = gl.arange(0, BN, layout=gl.SliceLayout(0, S_L))
    for i in range(ntiles):
        s = i % NS
        mbarrier.wait(ready.index(s), (i // NS) & 1)
        kt = k_ring.index(s)
        s0t = warpgroup_mma(q0, kt.slice(0, D, 1).permute((1, 0)), gl.zeros([M, BN], gl.float32, S_L),
                            use_acc=False, is_async=True)
        s1t = warpgroup_mma(q1, kt.slice(D, D, 1).permute((1, 0)), gl.zeros([M, BN], gl.float32, S_L),
                            use_acc=False, is_async=True)
        vt = v_ring.index(s)
        vh0.store(vt.slice(0, D, 1).load(VL).to(gl.float16))
        vh1.store(vt.slice(D, D, 1).load(VL).to(gl.float16))
        fence_async_shared()
        t = lo + i * BN + n_s
        s0, s1 = warpgroup_mma_wait(0, deps=[s0t, s1t])
        gl.barrier()
        mbarrier.arrive(empty.index(s))
        m0, l0, a0, p0 = _softmax(s0, t, hi, m0, l0, a0, scale, S_L, O_L, P_L)
        a0t = warpgroup_mma(p0, vh0, a0, is_async=True)
        m1, l1, a1, p1 = _softmax(s1, t, hi, m1, l1, a1, scale, S_L, O_L, P_L)
        a1t = warpgroup_mma(p1, vh1, a1, is_async=True)
        a0, a1, p0, p1 = warpgroup_mma_wait(0, deps=[a0t, a1t, p0, p1])
    return m0, l0, a0, m1, l1, a1


@gluon.jit
def _gws(qkv_ptr, kc_ptr, vc_ptr, ks_ptr, pt_ptr, seq_ptr, pacc_ptr, pml_ptr, qk_scale,
         HQ: gl.constexpr, HKV: gl.constexpr, G: gl.constexpr, L: gl.constexpr, D: gl.constexpr,
         MAX_PAGES: gl.constexpr, S: gl.constexpr, BN: gl.constexpr, NS: gl.constexpr,
         PW: gl.constexpr, PREGS: gl.constexpr):
    M: gl.constexpr = L * G
    C: gl.constexpr = 2 * D
    split = gl.program_id(0)
    b = gl.program_id(1)
    h = gl.program_id(2) * 2
    QKV_W: gl.constexpr = (2 * HQ + 2 * HKV) * D
    ROW_W: gl.constexpr = HKV * D
    QL: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    SH8: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=8, rank=2)
    SH16: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    VSH: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    SQ: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=64, element_bitwidth=8, rank=2)

    seq = gl.load(seq_ptr + b)
    chunk = gl.cdiv(gl.cdiv(seq, S), BN) * BN
    lo = split * chunk
    hi = gl.minimum(lo + chunk, seq)
    ntiles = gl.cdiv(gl.maximum(hi - lo, 0), BN)
    pt_row = pt_ptr + b.to(gl.int64) * MAX_PAGES

    k_ring = gl.allocate_shared_memory(gl.float8e4nv, [NS, BN, C], SH8)
    v_ring = gl.allocate_shared_memory(gl.float8e4nv, [NS, BN, C], VSH)
    vh0 = gl.allocate_shared_memory(gl.float16, [BN, D], SH16)
    vh1 = gl.allocate_shared_memory(gl.float16, [BN, D], SH16)
    ready = gl.allocate_shared_memory(gl.int64, [NS, 1], mbarrier.MBarrierLayout())
    empty = gl.allocate_shared_memory(gl.int64, [NS, 1], mbarrier.MBarrierLayout())
    for j in gl.static_range(NS):
        mbarrier.init(ready.index(j), count=PW * 32)
        mbarrier.init(empty.index(j), count=1)
    r = gl.arange(0, M, layout=gl.SliceLayout(1, QL))
    cq = gl.arange(0, D, layout=gl.SliceLayout(0, QL))
    q_off = (b * L + r // G)[:, None] * QKV_W + (h * G + r % G)[:, None] * D + cq[None, :]
    x0 = gl.load(qkv_ptr + q_off).to(gl.float32)
    x0 = x0 * gl.rsqrt(gl.sum(x0 * x0, 1) / D + 1e-5)[:, None]
    q0 = gl.allocate_shared_memory(gl.float8e4nv, [M, D], SQ, x0.to(gl.float8e4nv))
    x1 = gl.load(qkv_ptr + q_off + G * D).to(gl.float32)
    x1 = x1 * gl.rsqrt(gl.sum(x1 * x1, 1) / D + 1e-5)[:, None]
    q1 = gl.allocate_shared_memory(gl.float8e4nv, [M, D], SQ, x1.to(gl.float8e4nv))
    scale = qk_scale * gl.load(ks_ptr)
    fence_async_shared()
    gl.barrier()

    m0, l0, a0, m1, l1, a1 = gl.warp_specialize([
        (_consumer, (q0, q1, k_ring, v_ring, vh0, vh1, ready, empty, lo, hi, ntiles, scale, M, D, BN, NS)),
        (_producer, (pt_row, kc_ptr + h * D, vc_ptr + h * D, k_ring, v_ring, ready, empty, lo, hi, ntiles,
                     ROW_W, BN, C, NS, PW)),
    ], [PW], [PREGS])
    S_L: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, BN, 32])
    O_L: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, D, 16])
    _store_part(pacc_ptr, pml_ptr, b, h, split, m0, l0, a0, HKV, S, M, D, S_L, O_L)
    _store_part(pacc_ptr, pml_ptr, b, h + 1, split, m1, l1, a1, HKV, S, M, D, S_L, O_L)


@gluon.jit
def _consumer1(q, k_ring, v_ring, vh, ready, empty, lo, hi, ntiles, scale, pacc_ptr, pml_ptr, b, hh, split,
               HEAD: gl.constexpr, HKV: gl.constexpr, S: gl.constexpr,
               M: gl.constexpr, D: gl.constexpr, BN: gl.constexpr, NS: gl.constexpr):
    """One head's consumer warpgroup (FA3 style): K/V are the HEAD half of the shared 128-byte rows."""
    VL: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    S_L: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, BN, 32])
    O_L: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, D, 16])
    P_L: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=O_L, k_width=2)
    m0 = gl.full([M], -1.0e30, gl.float32, layout=gl.SliceLayout(1, S_L))
    l0 = gl.zeros([M], gl.float32, layout=gl.SliceLayout(1, S_L))
    a0 = gl.zeros([M, D], gl.float32, layout=O_L)
    n_s = gl.arange(0, BN, layout=gl.SliceLayout(0, S_L))
    for i in range(ntiles):
        s = i % NS
        mbarrier.wait(ready.index(s), (i // NS) & 1)
        s0t = warpgroup_mma(q, k_ring.index(s).slice(HEAD * D, D, 1).permute((1, 0)),
                            gl.zeros([M, BN], gl.float32, S_L), use_acc=False, is_async=True)
        vh.store(v_ring.index(s).slice(HEAD * D, D, 1).load(VL).to(gl.float16))
        fence_async_shared()
        t = lo + i * BN + n_s
        s0 = warpgroup_mma_wait(0, deps=[s0t])
        gl.barrier()
        mbarrier.arrive(empty.index(s))
        m0, l0, a0, p0 = _softmax(s0, t, hi, m0, l0, a0, scale, S_L, O_L, P_L)
        a0t = warpgroup_mma(p0, vh, a0, is_async=True)
        a0, p0 = warpgroup_mma_wait(0, deps=[a0t, p0])
    _store_part(pacc_ptr, pml_ptr, b, hh, split, m0, l0, a0, HKV, S, M, D, S_L, O_L)


@gluon.jit
def _gws2(qkv_ptr, kc_ptr, vc_ptr, ks_ptr, pt_ptr, seq_ptr, pacc_ptr, pml_ptr, qk_scale,
          HQ: gl.constexpr, HKV: gl.constexpr, G: gl.constexpr, L: gl.constexpr, D: gl.constexpr,
          MAX_PAGES: gl.constexpr, S: gl.constexpr, BN: gl.constexpr, NS: gl.constexpr,
          PW: gl.constexpr, PREGS: gl.constexpr, CREGS: gl.constexpr):
    M: gl.constexpr = L * G
    C: gl.constexpr = 2 * D
    split = gl.program_id(0)
    b = gl.program_id(1)
    h = gl.program_id(2) * 2
    QKV_W: gl.constexpr = (2 * HQ + 2 * HKV) * D
    ROW_W: gl.constexpr = HKV * D
    QL: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    SH8: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=8, rank=2)
    SH16: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    VSH: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    SQ: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=64, element_bitwidth=8, rank=2)

    seq = gl.load(seq_ptr + b)
    chunk = gl.cdiv(gl.cdiv(seq, S), BN) * BN
    lo = split * chunk
    hi = gl.minimum(lo + chunk, seq)
    ntiles = gl.cdiv(gl.maximum(hi - lo, 0), BN)
    pt_row = pt_ptr + b.to(gl.int64) * MAX_PAGES

    k_ring = gl.allocate_shared_memory(gl.float8e4nv, [NS, BN, C], SH8)
    v_ring = gl.allocate_shared_memory(gl.float8e4nv, [NS, BN, C], VSH)
    vh0 = gl.allocate_shared_memory(gl.float16, [BN, D], SH16)
    vh1 = gl.allocate_shared_memory(gl.float16, [BN, D], SH16)
    ready = gl.allocate_shared_memory(gl.int64, [NS, 1], mbarrier.MBarrierLayout())
    empty = gl.allocate_shared_memory(gl.int64, [NS, 1], mbarrier.MBarrierLayout())
    for j in gl.static_range(NS):
        mbarrier.init(ready.index(j), count=PW * 32)
        mbarrier.init(empty.index(j), count=2)
    r = gl.arange(0, M, layout=gl.SliceLayout(1, QL))
    cq = gl.arange(0, D, layout=gl.SliceLayout(0, QL))
    q_off = (b * L + r // G)[:, None] * QKV_W + (h * G + r % G)[:, None] * D + cq[None, :]
    x0 = gl.load(qkv_ptr + q_off).to(gl.float32)
    x0 = x0 * gl.rsqrt(gl.sum(x0 * x0, 1) / D + 1e-5)[:, None]
    q0 = gl.allocate_shared_memory(gl.float8e4nv, [M, D], SQ, x0.to(gl.float8e4nv))
    x1 = gl.load(qkv_ptr + q_off + G * D).to(gl.float32)
    x1 = x1 * gl.rsqrt(gl.sum(x1 * x1, 1) / D + 1e-5)[:, None]
    q1 = gl.allocate_shared_memory(gl.float8e4nv, [M, D], SQ, x1.to(gl.float8e4nv))
    scale = qk_scale * gl.load(ks_ptr)
    fence_async_shared()
    gl.barrier()

    gl.warp_specialize([
        (_consumer1, (q0, k_ring, v_ring, vh0, ready, empty, lo, hi, ntiles, scale, pacc_ptr, pml_ptr, b, h, split,
                      0, HKV, S, M, D, BN, NS)),
        (_consumer1, (q1, k_ring, v_ring, vh1, ready, empty, lo, hi, ntiles, scale, pacc_ptr, pml_ptr, b, h + 1,
                      split, 1, HKV, S, M, D, BN, NS)),
        (_producer, (pt_row, kc_ptr + h * D, vc_ptr + h * D, k_ring, v_ring, ready, empty, lo, hi, ntiles,
                     ROW_W, BN, C, NS, PW)),
    ], [4, PW], [CREGS, PREGS])


def run_ws(data, S, BN=32, NS=4, maxnreg=None, pw=1, pregs=24, cregs=0):
    cfg = data["config"]
    batch, length, hq, hkv, d = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    g = hq // hkv
    rows = length * g
    pacc = torch.empty((batch, hkv, S, rows, d), dtype=torch.float32, device="cuda")
    pml = torch.empty((batch, hkv, S, rows, 2), dtype=torch.float32, device="cuda")
    pt = data["page_table"]

    def fn():
        kern = _gws2 if cregs else _gws
        kw = dict(CREGS=cregs) if cregs else {}
        kern[(S, batch, hkv // 2)](data["qkv"], data["k_cache"], data["v_cache"], data["k_scale"], pt,
                                   data["seq_lens"], pacc, pml,
                                   cfg["sm_scale"] * cfg["qk_scale_factor"] * 1.4426950408889634,
                                   HQ=hq, HKV=hkv, G=g, L=length, D=d, MAX_PAGES=pt.shape[1], S=S, BN=BN, NS=NS,
                                   PW=pw, PREGS=pregs, **kw, num_warps=4, **({'maxnreg': maxnreg} if maxnreg else {}))
    return fn, pacc, pml


if __name__ == "__main__":
    import bench, submission
    load = sys.argv[1] if len(sys.argv) > 1 else "verify 16"
    c = next(c for l, c in bench.SCORED if l == load and c["layer"] == "full")
    data = bench.data_for(load, c)
    cap = triton_partials(data)
    a = [int(x) for x in sys.argv[2:]] + [0] * 8
    S = a[0] or cap["S"]
    NS = a[1] or 4
    mr = a[2] or None
    pw = a[3] or 1
    pregs = a[4] or 24
    cregs = a[5]
    fn, pacc, pml = run_ws(data, S, BN=a[6] or 32, NS=NS, maxnreg=mr, pw=pw, pregs=pregs, cregs=cregs)
    fn()
    torch.cuda.synchronize()
    import re
    for cache in (_gws2 if cregs else _gws).device_caches.values():
        for ck in cache[0].values():
            print("regs", ck.n_regs, "spills", ck.n_spills, "smem", ck.metadata.shared,
                  re.findall(r"setmaxnreg\S* \t?(\d+)", ck.asm["ptx"]), re.findall(r"\.reqntid \d+", ck.asm["ptx"]))
    if S == cap["S"]:
        ra = merged(cap["pacc"], cap["pml"], 1)
        rb = merged(pacc, pml, 1)
        print("max abs diff (splits>=1, merged):", (ra - rb).abs().max().item(), "ref max", ra.abs().max().item())
    # Independent of split boundaries: all cached keys merged, vs the BN=32 single-partition kernel (which is
    # bit-exact with Triton's partials for splits >= 1).
    from gluon_verify import run_gluon2
    rfn, rpacc, rpml = run_gluon2(data, S, BN=32, NS=4, maxnreg=128, pf=True, vhb=1)
    rfn()
    ra, rb = merged(rpacc, rpml, 0), merged(pacc, pml, 0)
    print("all-keys max abs diff vs gluon BN32:", (ra - rb).abs().max().item(),
          "rel l2", ((ra - rb).norm() / ra.norm()).item())
    t_g = time_fn(fn)
    t_t = time_fn(lambda: submission.kernel_fn(data))
    print(f"{load} full: gluon-ws partials {t_g:.1f} us (S={S}, NS={NS}, maxnreg={mr}, pw={pw}, pregs={pregs}, cregs={cregs}); "
          f"triton kernel_fn {t_t:.1f} us")

"""Prototype: verify-full cached-key attention in Gluon (explicit cp.async ring, async wgmma).

Writes the same partials as submission._attn_split (m, l in log2 units, unnormalized acc) for the
cached keys only (no new tokens). Compares against the Triton kernel's partials for splits >= 1
(split 0 also holds the new tokens there) and times both.

  python tools/gluon_verify.py "verify 16" S NS MODE MAXNREG BN PF VHB
    MODE: serial (single head, default) | pipe | defer | pf | pf2 | pf3 | dual (_gv2, both heads)
  Best so far (H100): python tools/gluon_verify.py "verify 16" 33 4 dual 128 32 pf 1  -> 154 us
                      python tools/gluon_verify.py "verify 32" 16 4 dual 128 32 pf 1  -> 304 us
  (Triton main: 155/294 us for the whole kernel_fn, ~149/287 attn only.)
"""
import fcntl, statistics, sys
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import (fence_async_shared, warpgroup_mma, warpgroup_mma_wait,
                                                                warpgroup_mma_init)
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp

sys.path.insert(0, ".")
sys.path.insert(0, "tools")


@gluon.jit
def _issue(i, lo, hi, pt_row, kc_ptr, vc_ptr, k_ring, v_ring, h, n_g, c_g,
           ROW_W: gl.constexpr, D: gl.constexpr, BN: gl.constexpr, NS: gl.constexpr):
    idx = gl.minimum(lo + i * BN + n_g, hi - 1)
    page = gl.load(pt_row + idx)
    row = page.to(gl.int64) * ROW_W + h * D
    off = row[:, None] + c_g[None, :]
    cp.async_copy_global_to_shared(k_ring.index(i % NS), kc_ptr + off)
    cp.async_copy_global_to_shared(v_ring.index(i % NS), vc_ptr + off)
    cp.commit_group()


@gluon.jit
def _pages(i, lo, hi, pt_row, n_g, BN: gl.constexpr):
    return gl.load(pt_row + gl.minimum(lo + i * BN + n_g, hi - 1))


@gluon.jit
def _issue_pg(i, page, kc_ptr, vc_ptr, k_ring, v_ring, h, c_g, ROW_W: gl.constexpr, D: gl.constexpr,
              NS: gl.constexpr):
    off = (page.to(gl.int64) * ROW_W + h * D)[:, None] + c_g[None, :]
    cp.async_copy_global_to_shared(k_ring.index(i % NS), kc_ptr + off)
    cp.async_copy_global_to_shared(v_ring.index(i % NS), vc_ptr + off)
    cp.commit_group()


@gluon.jit
def _gv(qkv_ptr, kc_ptr, vc_ptr, ks_ptr, pt_ptr, seq_ptr, pacc_ptr, pml_ptr, qk_scale,
        HQ: gl.constexpr, HKV: gl.constexpr, G: gl.constexpr, L: gl.constexpr, D: gl.constexpr,
        MAX_PAGES: gl.constexpr, S: gl.constexpr, BN: gl.constexpr, NS: gl.constexpr,
        PIPE: gl.constexpr = False, DEFER: gl.constexpr = False,
        PF: gl.constexpr = 0):
    M: gl.constexpr = L * G
    split = gl.program_id(0)
    b = gl.program_id(1)
    h = gl.program_id(2)
    QKV_W: gl.constexpr = (2 * HQ + 2 * HKV) * D
    ROW_W: gl.constexpr = HKV * D
    QL: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    GLY: gl.constexpr = gl.BlockedLayout([1, 16], [8, 4], [4, 1], [1, 0])
    VL: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    S_L: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, BN, 32])
    O_L: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, D, 16])
    P_L: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=O_L, k_width=2)
    SH8: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=64, element_bitwidth=8, rank=2)
    SH16: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)

    seq = gl.load(seq_ptr + b)
    chunk = gl.cdiv(gl.cdiv(seq, S), BN) * BN
    lo = split * chunk
    hi = gl.minimum(lo + chunk, seq)
    ntiles = gl.cdiv(gl.maximum(hi - lo, 0), BN)
    pt_row = pt_ptr + b.to(gl.int64) * MAX_PAGES

    k_ring = gl.allocate_shared_memory(gl.float8e4nv, [NS, BN, D], SH8)
    v_ring = gl.allocate_shared_memory(gl.float8e4nv, [NS, BN, D], SH8)
    vh = gl.allocate_shared_memory(gl.float16, [2, BN, D], SH16)
    n_g = gl.arange(0, BN, layout=gl.SliceLayout(1, GLY))
    c_g = gl.arange(0, D, layout=gl.SliceLayout(0, GLY))
    for i in gl.static_range(NS - 1):
        _issue(i, lo, hi, pt_row, kc_ptr, vc_ptr, k_ring, v_ring, h, n_g, c_g, ROW_W, D, BN, NS)

    r = gl.arange(0, M, layout=gl.SliceLayout(1, QL))
    cq = gl.arange(0, D, layout=gl.SliceLayout(0, QL))
    q_off = (b * L + r // G)[:, None] * QKV_W + (h * G + r % G)[:, None] * D + cq[None, :]
    x = gl.load(qkv_ptr + q_off).to(gl.float32)
    x = x * gl.rsqrt(gl.sum(x * x, 1) / D + 1e-5)[:, None]
    q_smem = gl.allocate_shared_memory(gl.float8e4nv, [M, D], SH8, x.to(gl.float8e4nv))
    scale = qk_scale * gl.load(ks_ptr)

    m_i = gl.full([M], -1.0e30, gl.float32, layout=gl.SliceLayout(1, S_L))
    l_i = gl.zeros([M], gl.float32, layout=gl.SliceLayout(1, S_L))
    acc = gl.zeros([M, D], gl.float32, layout=O_L)
    n_s = gl.arange(0, BN, layout=gl.SliceLayout(0, S_L))
    if PIPE:
        # FA3-style: QK^T of tile i+1 runs on the tensor cores while tile i goes through the softmax.
        cp.wait_group(NS - 2)
        fence_async_shared()
        gl.barrier()
        s_tok = warpgroup_mma(q_smem, k_ring.index(0).permute((1, 0)), gl.zeros([M, BN], gl.float32, S_L),
                              use_acc=False, is_async=True)
        acc_tok = warpgroup_mma_init(acc)
        for i in range(ntiles):
            cp.wait_group(NS - 3)
            fence_async_shared()
            gl.barrier()
            s_next = warpgroup_mma(q_smem, k_ring.index((i + 1) % NS).permute((1, 0)),
                                   gl.zeros([M, BN], gl.float32, S_L), use_acc=False, is_async=True)
            _issue(i + NS - 1, lo, hi, pt_row, kc_ptr, vc_ptr, k_ring, v_ring, h, n_g, c_g, ROW_W, D, BN, NS)
            v8 = v_ring.index(i % NS).load(VL)
            vh.index(i % 2).store(v8.to(gl.float16))
            fence_async_shared()
            s, acc = warpgroup_mma_wait(1, deps=[s_tok, acc_tok])
            t = lo + i * BN + n_s
            s = gl.where((t < hi)[None, :], s, float("-inf"))
            m_new = gl.maximum(m_i, gl.max(s, 1) * scale)
            alpha = gl.exp2(m_i - m_new)
            p = gl.exp2(gl.fma(s, scale, -m_new[:, None]))
            l_i = l_i * alpha + gl.sum(p, 1)
            m_i = m_new
            acc = acc * gl.convert_layout(alpha, gl.SliceLayout(1, O_L))[:, None]
            p16 = gl.convert_layout(p.to(gl.float16), P_L)
            gl.barrier()
            acc_tok = warpgroup_mma(p16, vh.index(i % 2), acc, is_async=True)
            s_tok = s_next
        acc, s = warpgroup_mma_wait(0, deps=[acc_tok, s_tok])
    elif DEFER:
        # PV of tile i is waited for only after QK^T of tile i+1 has been issued.
        acc_tok = warpgroup_mma_init(acc)
        for i in range(ntiles):
            cp.wait_group(NS - 2)
            fence_async_shared()
            gl.barrier()
            s_tok = warpgroup_mma(q_smem, k_ring.index(i % NS).permute((1, 0)), gl.zeros([M, BN], gl.float32, S_L),
                                  use_acc=False, is_async=True)
            v8 = v_ring.index(i % NS).load(VL)
            vh.index(i % 2).store(v8.to(gl.float16))
            fence_async_shared()
            _issue(i + NS - 1, lo, hi, pt_row, kc_ptr, vc_ptr, k_ring, v_ring, h, n_g, c_g, ROW_W, D, BN, NS)
            s, acc = warpgroup_mma_wait(0, deps=[s_tok, acc_tok])
            t = lo + i * BN + n_s
            s = gl.where((t < hi)[None, :], s, float("-inf"))
            m_new = gl.maximum(m_i, gl.max(s, 1) * scale)
            alpha = gl.exp2(m_i - m_new)
            p = gl.exp2(gl.fma(s, scale, -m_new[:, None]))
            l_i = l_i * alpha + gl.sum(p, 1)
            m_i = m_new
            acc = acc * gl.convert_layout(alpha, gl.SliceLayout(1, O_L))[:, None]
            p16 = gl.convert_layout(p.to(gl.float16), P_L)
            gl.barrier()
            acc_tok = warpgroup_mma(p16, vh.index(i % 2), acc, is_async=True)
        acc = warpgroup_mma_wait(0, deps=[acc_tok])
    elif PF:
        # Page indices of tile i + NS - 1 were loaded one iteration earlier.
        pg = _pages(NS - 1, lo, hi, pt_row, n_g, BN)
        pg2 = _pages(NS, lo, hi, pt_row, n_g, BN)
        pg3 = _pages(NS + 1, lo, hi, pt_row, n_g, BN)
        for i in range(ntiles):
            cp.wait_group(NS - 2)
            fence_async_shared()
            gl.barrier()
            s_tok = warpgroup_mma(q_smem, k_ring.index(i % NS).permute((1, 0)), gl.zeros([M, BN], gl.float32, S_L),
                                  use_acc=False, is_async=True)
            v8 = v_ring.index(i % NS).load(VL)
            vh.index(i % 2).store(v8.to(gl.float16))
            fence_async_shared()
            _issue_pg(i + NS - 1, pg, kc_ptr, vc_ptr, k_ring, v_ring, h, c_g, ROW_W, D, NS)
            if PF == 1:
                pg = _pages(i + NS, lo, hi, pt_row, n_g, BN)
            elif PF == 2:
                pg = pg2
                pg2 = _pages(i + NS + 1, lo, hi, pt_row, n_g, BN)
            else:
                pg = pg2
                pg2 = pg3
                pg3 = _pages(i + NS + 2, lo, hi, pt_row, n_g, BN)
            s = warpgroup_mma_wait(0, deps=[s_tok])
            t = lo + i * BN + n_s
            m_i, l_i, acc, p16 = _softmax(s, t, hi, m_i, l_i, acc, scale, S_L, O_L, P_L)
            gl.barrier()
            acc_tok = warpgroup_mma(p16, vh.index(i % 2), acc, is_async=True)
            acc, p16 = warpgroup_mma_wait(0, deps=[acc_tok, p16])
    else:
        for i in range(ntiles):
            cp.wait_group(NS - 2)
            fence_async_shared()
            gl.barrier()
            s_tok = warpgroup_mma(q_smem, k_ring.index(i % NS).permute((1, 0)), gl.zeros([M, BN], gl.float32, S_L),
                                  use_acc=False, is_async=True)
            v8 = v_ring.index(i % NS).load(VL)
            vh.index(i % 2).store(v8.to(gl.float16))
            fence_async_shared()
            _issue(i + NS - 1, lo, hi, pt_row, kc_ptr, vc_ptr, k_ring, v_ring, h, n_g, c_g, ROW_W, D, BN, NS)
            s = warpgroup_mma_wait(0, deps=[s_tok])
            t = lo + i * BN + n_s
            m_i, l_i, acc, p16 = _softmax(s, t, hi, m_i, l_i, acc, scale, S_L, O_L, P_L)
            gl.barrier()
            acc_tok = warpgroup_mma(p16, vh.index(i % 2), acc, is_async=True)
            acc, p16 = warpgroup_mma_wait(0, deps=[acc_tok, p16])
    cp.wait_group(0)

    ro = gl.arange(0, M, layout=gl.SliceLayout(1, O_L))
    co = gl.arange(0, D, layout=gl.SliceLayout(0, O_L))
    part = ((b * HKV + h) * S + split) * M + ro
    gl.store(pacc_ptr + part[:, None] * D + co[None, :], acc)
    rs = gl.arange(0, M, layout=gl.SliceLayout(1, S_L))
    parts = ((b * HKV + h) * S + split) * M + rs
    gl.store(pml_ptr + parts * 2, m_i)
    gl.store(pml_ptr + parts * 2 + 1, l_i)


@gluon.jit
def _softmax(s, t, hi, m_i, l_i, acc, scale, S_L: gl.constexpr, O_L: gl.constexpr, P_L: gl.constexpr):
    s = gl.where((t < hi)[None, :], s, float("-inf"))
    m_new = gl.maximum(m_i, gl.max(s, 1) * scale)
    alpha = gl.exp2(m_i - m_new)
    p = gl.exp2(gl.fma(s, scale, -m_new[:, None]))
    l_i = l_i * alpha + gl.sum(p, 1)
    acc = acc * gl.convert_layout(alpha, gl.SliceLayout(1, O_L))[:, None]
    return m_new, l_i, acc, gl.convert_layout(p.to(gl.float16), P_L)


@gluon.jit
def _issue2(i, lo, hi, pt_row, kc_ptr, vc_ptr, k_ring, v_ring, n_g, c_g,
            ROW_W: gl.constexpr, BN: gl.constexpr, NS: gl.constexpr):
    idx = gl.minimum(lo + i * BN + n_g, hi - 1)
    page = gl.load(pt_row + idx)
    off = (page.to(gl.int64) * ROW_W)[:, None] + c_g[None, :]
    cp.async_copy_global_to_shared(k_ring.index(i % NS), kc_ptr + off)
    cp.async_copy_global_to_shared(v_ring.index(i % NS), vc_ptr + off)
    cp.commit_group()


@gluon.jit
def _store_part(pacc_ptr, pml_ptr, b, hh, split, m_i, l_i, acc, HKV: gl.constexpr, S: gl.constexpr,
                M: gl.constexpr, D: gl.constexpr, S_L: gl.constexpr, O_L: gl.constexpr):
    ro = gl.arange(0, M, layout=gl.SliceLayout(1, O_L))
    co = gl.arange(0, D, layout=gl.SliceLayout(0, O_L))
    part = ((b * HKV + hh) * S + split) * M + ro
    gl.store(pacc_ptr + part[:, None] * D + co[None, :], acc)
    rs = gl.arange(0, M, layout=gl.SliceLayout(1, S_L))
    parts = ((b * HKV + hh) * S + split) * M + rs
    gl.store(pml_ptr + parts * 2, m_i)
    gl.store(pml_ptr + parts * 2 + 1, l_i)


@gluon.jit
def _gv2(qkv_ptr, kc_ptr, vc_ptr, ks_ptr, pt_ptr, seq_ptr, pacc_ptr, pml_ptr, qk_scale,
         HQ: gl.constexpr, HKV: gl.constexpr, G: gl.constexpr, L: gl.constexpr, D: gl.constexpr,
         MAX_PAGES: gl.constexpr, S: gl.constexpr, BN: gl.constexpr, NS: gl.constexpr,
         PF: gl.constexpr = False, VHB: gl.constexpr = 2):
    """Both KV heads of a pair per program: one 128-byte row gather per key for K and for V, both
    heads' QK^T against the same [BN, 2D] K tile (queries zero outside their head's D columns),
    two independent softmax chains."""
    M: gl.constexpr = L * G
    C: gl.constexpr = 2 * D
    split = gl.program_id(0)
    b = gl.program_id(1)
    h = gl.program_id(2) * 2
    QKV_W: gl.constexpr = (2 * HQ + 2 * HKV) * D
    ROW_W: gl.constexpr = HKV * D
    QL: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    GLY: gl.constexpr = gl.BlockedLayout([1, 16], [4, 8], [4, 1], [1, 0])
    VL: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    S_L: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, BN, 32])
    O_L: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, D, 16])
    P_L: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=O_L, k_width=2)
    SH8: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=8, rank=2)
    SH16: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    VSH: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    SQ: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=64, element_bitwidth=8, rank=2)

    seq = gl.load(seq_ptr + b)
    chunk = gl.cdiv(gl.cdiv(seq, S), BN) * BN
    lo = split * chunk
    hi = gl.minimum(lo + chunk, seq)
    ntiles = gl.cdiv(gl.maximum(hi - lo, 0), BN)
    pt_row = pt_ptr + b.to(gl.int64) * MAX_PAGES + h * 0

    k_ring = gl.allocate_shared_memory(gl.float8e4nv, [NS, BN, C], SH8)
    v_ring = gl.allocate_shared_memory(gl.float8e4nv, [NS, BN, C], VSH)
    vh0 = gl.allocate_shared_memory(gl.float16, [VHB, BN, D], SH16)
    vh1 = gl.allocate_shared_memory(gl.float16, [VHB, BN, D], SH16)
    n_g = gl.arange(0, BN, layout=gl.SliceLayout(1, GLY))
    c_g = gl.arange(0, C, layout=gl.SliceLayout(0, GLY))
    kbase = kc_ptr + h * D
    vbase = vc_ptr + h * D
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
    for i in gl.static_range(NS - 1):
        _issue2(i, lo, hi, pt_row, kbase, vbase, k_ring, v_ring, n_g, c_g, ROW_W, BN, NS)


    m0 = gl.full([M], -1.0e30, gl.float32, layout=gl.SliceLayout(1, S_L))
    l0 = gl.zeros([M], gl.float32, layout=gl.SliceLayout(1, S_L))
    a0 = gl.zeros([M, D], gl.float32, layout=O_L)
    m1, l1, a1 = m0, l0, a0
    n_s = gl.arange(0, BN, layout=gl.SliceLayout(0, S_L))
    pg = _pages(NS - 1, lo, hi, pt_row, n_g, BN)
    pg2 = _pages(NS, lo, hi, pt_row, n_g, BN)
    for i in range(ntiles):
        cp.wait_group(NS - 2)
        fence_async_shared()
        gl.barrier()
        kt = k_ring.index(i % NS)
        s0t = warpgroup_mma(q0, kt.slice(0, D, 1).permute((1, 0)), gl.zeros([M, BN], gl.float32, S_L),
                            use_acc=False, is_async=True)
        s1t = warpgroup_mma(q1, kt.slice(D, D, 1).permute((1, 0)), gl.zeros([M, BN], gl.float32, S_L),
                            use_acc=False, is_async=True)
        vt = v_ring.index(i % NS)
        vh0.index(i % VHB).store(vt.slice(0, D, 1).load(VL).to(gl.float16))
        vh1.index(i % VHB).store(vt.slice(D, D, 1).load(VL).to(gl.float16))
        fence_async_shared()
        gl.barrier()
        if PF:
            _issue_pg(i + NS - 1, pg, kbase, vbase, k_ring, v_ring, 0, c_g, ROW_W, D, NS)
            pg = pg2
            pg2 = _pages(i + NS + 1, lo, hi, pt_row, n_g, BN)
        else:
            _issue2(i + NS - 1, lo, hi, pt_row, kbase, vbase, k_ring, v_ring, n_g, c_g, ROW_W, BN, NS)
        t = lo + i * BN + n_s
        s0 = warpgroup_mma_wait(1, deps=[s0t])
        m0, l0, a0, p0 = _softmax(s0, t, hi, m0, l0, a0, scale, S_L, O_L, P_L)
        a0t = warpgroup_mma(p0, vh0.index(i % VHB), a0, is_async=True)
        s1 = warpgroup_mma_wait(1, deps=[s1t])
        m1, l1, a1, p1 = _softmax(s1, t, hi, m1, l1, a1, scale, S_L, O_L, P_L)
        a1t = warpgroup_mma(p1, vh1.index(i % VHB), a1, is_async=True)
        a0, a1, p0, p1 = warpgroup_mma_wait(0, deps=[a0t, a1t, p0, p1])
    cp.wait_group(0)
    _store_part(pacc_ptr, pml_ptr, b, h, split, m0, l0, a0, HKV, S, M, D, S_L, O_L)
    _store_part(pacc_ptr, pml_ptr, b, h + 1, split, m1, l1, a1, HKV, S, M, D, S_L, O_L)




def run_gluon2(data, S, BN=32, NS=4, warps=4, maxnreg=None, pf=False, vhb=2):
    cfg = data["config"]
    batch, length, hq, hkv, d = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    g = hq // hkv
    rows = length * g
    pacc = torch.empty((batch, hkv, S, rows, d), dtype=torch.float32, device="cuda")
    pml = torch.empty((batch, hkv, S, rows, 2), dtype=torch.float32, device="cuda")
    pt = data["page_table"]

    def fn():
        _gv2[(S, batch, hkv // 2)](data["qkv"], data["k_cache"], data["v_cache"], data["k_scale"], pt,
                                   data["seq_lens"], pacc, pml,
                                   cfg["sm_scale"] * cfg["qk_scale_factor"] * 1.4426950408889634,
                                   HQ=hq, HKV=hkv, G=g, L=length, D=d, MAX_PAGES=pt.shape[1], S=S, BN=BN, NS=NS, PF=pf, VHB=vhb,
                                   num_warps=warps, **({'maxnreg': maxnreg} if maxnreg else {}))
    return fn, pacc, pml


def run_gluon(data, S, BN=32, NS=4, warps=4, pipe=False, maxnreg=None):
    cfg = data["config"]
    batch, length, hq, hkv, d = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    g = hq // hkv
    rows = length * g
    pacc = torch.empty((batch, hkv, S, rows, d), dtype=torch.float32, device="cuda")
    pml = torch.empty((batch, hkv, S, rows, 2), dtype=torch.float32, device="cuda")
    pt = data["page_table"]

    def fn():
        _gv[(S, batch, hkv)](data["qkv"], data["k_cache"], data["v_cache"], data["k_scale"], pt, data["seq_lens"],
                             pacc, pml, cfg["sm_scale"] * cfg["qk_scale_factor"] * 1.4426950408889634,
                             HQ=hq, HKV=hkv, G=g, L=length, D=d, MAX_PAGES=pt.shape[1], S=S, BN=BN, NS=NS, PIPE=pipe == True, DEFER=pipe == "defer", PF={"pf": 1, "pf2": 2, "pf3": 3}.get(pipe, 0) if isinstance(pipe, str) else 0,
                             num_warps=warps, **({'maxnreg': maxnreg} if maxnreg else {}))
    return fn, pacc, pml


def triton_partials(data):
    import submission
    cap = {}
    orig = submission._combine

    class Grab:
        def __getitem__(self, grid):
            def f(*a, **k):
                cap["pacc"], cap["pml"], cap["S"] = a[1], a[2], k["S"]
            return f
    submission._combine = Grab()
    try:
        submission.kernel_fn(data)
    finally:
        submission._combine = orig
    return cap


def merged(pacc, pml, first):
    """Merge splits first.. of [B, H, S, M, D] partials into [B, H, M, D] (normalized)."""
    m = pml[:, :, first:, :, 0]
    l = pml[:, :, first:, :, 1]
    mm = m.max(2, keepdim=True).values
    w = torch.exp2(m - mm)
    return (w[..., None] * pacc[:, :, first:]).sum(2) / (w * l).sum(2)[..., None]


def time_fn(fn, reps=40):
    import bench
    return bench.time_graph(fn, reps) - bench.empty_cost()


if __name__ == "__main__":
    import bench, submission
    load = sys.argv[1] if len(sys.argv) > 1 else "verify 16"
    c = next(c for l, c in bench.SCORED if l == load and c["layer"] == "full")
    data = bench.data_for(load, c)
    cap = triton_partials(data)
    S = int(sys.argv[2]) if len(sys.argv) > 2 else cap["S"]
    NS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    pipe = {"pipe": True, "defer": "defer", "pf": "pf", "pf2": "pf2", "pf3": "pf3"}.get(sys.argv[4] if len(sys.argv) > 4 else "", False)
    mr = (int(sys.argv[5]) or None) if len(sys.argv) > 5 else None
    if pipe == "dual" or (len(sys.argv) > 4 and sys.argv[4] == "dual"):
        fn, pacc, pml = run_gluon2(data, S, BN=int(sys.argv[6]) if len(sys.argv) > 6 else 32, NS=NS, maxnreg=mr, pf=len(sys.argv) > 7 and sys.argv[7] == "pf",
                                  vhb=int(sys.argv[8]) if len(sys.argv) > 8 else 2)
    else:
        BN = int(sys.argv[6]) if len(sys.argv) > 6 else 32
        fn, pacc, pml = run_gluon(data, S, BN=BN, NS=NS, pipe=pipe, maxnreg=mr)
    fn()
    torch.cuda.synchronize()
    for kern in (_gv, _gv2):
        for cache in kern.device_caches.values():
            for ck in cache[0].values():
                print(kern.__name__, "regs", ck.n_regs, "spills", ck.n_spills, "smem", ck.metadata.shared)
    # Compare cached-key attention excluding the new tokens: triton splits >= 1 vs gluon. Splits differ in
    # count, so compare the merge over keys >= triton's split-1 start? Simplest: when S matches, compare
    # splits >= 1 merged.
    if S == cap["S"]:
        a = merged(cap["pacc"], cap["pml"], 1)
        bq = merged(pacc, pml, 1)
        print("max abs diff (splits>=1, merged):", (a - bq).abs().max().item(), "ref max", a.abs().max().item())
    t_g = time_fn(fn)
    t_t = time_fn(lambda: submission.kernel_fn(data))
    print(f"{load} full: gluon partials {t_g:.1f} us (S={S}, NS={NS}, pipe={pipe}, maxnreg={mr}); triton full kernel_fn {t_t:.1f} us")

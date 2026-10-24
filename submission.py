# okc: problem=muse_attention_decode
"""Gated attention decode in two Triton launches.

1. _attn_split: one program per (split of the context, request, KV head). It normalizes (and
   on sliding layers ropes) its group's queries straight from the fused projection output,
   streams its slice of the paged FP8 cache through an online softmax, and writes a partial
   (max, sum, accumulator). The split-0 program also normalizes/ropes the block's new keys,
   quantizes the new keys and values to FP8, stores them in the cache, and attends them
   causally from registers.
2. _combine: one program per (token, KV head). It merges the splits, then applies the
   per-head RMSNorm and sigmoid(gate)."""

from typing import TypedDict

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents, gdc_wait

try:  # Gluon (Triton's low-level dialect, ships with triton 3.8): the verify full-layer kernel.
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.language.nvidia.hopper import (fence_async_shared, mbarrier, warpgroup_mma,
                                                                    warpgroup_mma_wait)
    from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp
    from triton.experimental.gluon.language.nvidia.ampere import mma_v2
    HAVE_GLUON = True
except ImportError:
    HAVE_GLUON = False


class Config(TypedDict):
    batch: int            # B
    q_len: int            # L: 1 for decode, 8 for a speculative verify block (causal inside it)
    num_q_heads: int      # Hq: 16 per GPU under TP8
    num_kv_heads: int     # Hkv: 2; each serves Hq / Hkv query heads
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers; 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)


class Inputs(TypedDict):
    qkv: torch.Tensor          # bfloat16 [B, L, (2*Hq + 2*Hkv)*64], 2304 under tp8: per token q (Hq*64),
                               # then the block's keys before norm and RoPE (Hkv*64), values
                               # (Hkv*64) and the output gate's logits (Hq*64)
    positions: torch.Tensor    # int64 [B*L], each new token's position (seq_lens[b] + i)
    cache_loc: torch.Tensor    # int64 [B*L], each new token's flat cache row (page * page_size + offset)
    cache_seqlens: torch.Tensor  # int32 [B], seq_lens + L: what each request attends over
    cu_seqlens_q: torch.Tensor   # int32 [B+1], 0, L, 2L, ...: the block's query offsets
    scheduler_metadata: torch.Tensor | None  # int32, FA3's tile schedule (get_scheduler_metadata) on
                                             # full-attention decode steps, else None
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64], written in place
    k_scale: torch.Tensor     # float32 [], key = e4m3 * k_scale
    v_scale: torch.Tensor     # float32 [], value = e4m3 * v_scale
    page_table: torch.Tensor  # int32 [B, max_pages], physical page of each logical page (-1 unused)
    seq_lens: torch.Tensor    # int32 [B], cached tokens per request; the block starts at this position
    config: Config


RMS_EPS = 1e-5
LOG2E = 1.4426950408889634
ROPE_POSITIONS = 1 << 21
_ROPE = {}
_SMS = {}
_COUNTERS = {}
# The MMA input precision for P@V (and QK^T without qk8). FP16 holds every E4M3 value exactly and
# converts from it in one instruction (BF16 takes two); P in [0, 1] keeps more bits than in BF16.
# A test hook too: Triton's CPU interpreter can't multiply 16-bit floats.
DOT_DTYPE = tl.float16
# Programmatic dependent launch of the merge kernel (off in the CPU interpreter).
USE_PDL = True
USE_GDEC = True  # decode full layers at batch * hkv > 2 on the Gluon kernel (_gd_full)
USE_GSLIDE = False  # batched sliding decode on _gd_full too (H100 win, H200 loss)
USE_GD1 = True  # decode full at batch * hkv == 2 on _gd_full too (511 splits, 8-warp one-pass merge); unchecked on H200
USE_GBAL = True  # verify full from 16 KV head pairs: balanced splits over the whole batch (_gw_bal); unchecked on H200
# Split partials' accumulator dtype (plan key pdt).
_PDT = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}
print(f"env: torch {torch.__version__}, triton {triton.__version__}", flush=True)


def _cos_sin(theta, d, device):
    """[2^21, d] float32 cos | sin, built as SGLang's RotaryEmbedding (and the reference) do."""
    cache = _ROPE.get(theta)
    if cache is None:
        inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d))
        ang = torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device)[:, None] * inv_freq
        cache = _ROPE[theta] = torch.cat((ang.cos(), ang.sin()), dim=-1).contiguous()
    return cache


def _counters(n, device, tag=""):
    """Zeroed int32 arrival counters for the fused merges; each call leaves them zeroed again (the Gluon
    merge's generation words, tag "g", just count up)."""
    buf = _COUNTERS.get((device, tag))
    if buf is None or buf.numel() < n:
        buf = _COUNTERS[(device, tag)] = torch.zeros(max(n, 4096), dtype=torch.int32, device=device)
    return buf


def _num_sms(device):
    if device.type != "cuda":
        return 132
    if device.index not in _SMS:
        _SMS[device.index] = torch.cuda.get_device_properties(device).multi_processor_count
    return _SMS[device.index]


# ---- Verify blocks on full layers (Gluon, warp-specialized) ----
# One program per (split, request, KV head pair), FA3 style: a 4-warp producer gathers the pair's
# 128-byte K and V rows (page indices, cp.async) into an NS-slot shared ring and signals `ready`
# (cp.async.mbarrier.arrive); one consumer warpgroup per KV head runs QK^T (FP8 wgmma on its 64-byte
# half of each row), stages its V half as FP16, frees the slot (`empty`), and does the online softmax
# and P@V. The extra split S holds the block's new tokens (normalized, stored as FP8, attended causally).
if HAVE_GLUON:
    @gluon.jit
    def _gw_producer(pt_row, kbase, vbase, k_ring, v_ring, ready, empty, lo, hi, ntiles,
                     PAGE: gl.constexpr, ROW_W: gl.constexpr, BN: gl.constexpr, C: gl.constexpr,
                     NS: gl.constexpr, PW: gl.constexpr):
        GLY: gl.constexpr = gl.BlockedLayout([1, 16], [4, 8], [PW, 1], [1, 0])
        n_g = gl.arange(0, BN, layout=gl.SliceLayout(1, GLY))
        c_g = gl.arange(0, C, layout=gl.SliceLayout(0, GLY))
        t = gl.maximum(gl.minimum(lo + n_g, hi - 1), 0)
        pg = gl.load(pt_row + t // PAGE)
        for i in range(ntiles):
            s = i % NS
            off = ((pg.to(gl.int64) * PAGE + t % PAGE) * ROW_W)[:, None] + c_g[None, :]
            # Next tile's page indices before waiting for the slot.
            t = gl.maximum(gl.minimum(lo + (i + 1) * BN + n_g, hi - 1), 0)
            pg = gl.load(pt_row + t // PAGE)
            mbarrier.wait(empty.index(s), ((i // NS) & 1) ^ 1)
            cp.async_copy_global_to_shared(k_ring.index(s), kbase + off)
            cp.async_copy_global_to_shared(v_ring.index(s), vbase + off)
            cp.mbarrier_arrive(ready.index(s), increment_count=False)

    @gluon.jit
    def _gw_store(pacc_ptr, pml_ptr, b, hh, split, m_i, l_i, acc, HKV: gl.constexpr, SP: gl.constexpr,
                  M: gl.constexpr, D: gl.constexpr, R_L: gl.constexpr, A_L: gl.constexpr):
        ro = gl.arange(0, M, layout=gl.SliceLayout(1, A_L))
        co = gl.arange(0, D, layout=gl.SliceLayout(0, A_L))
        part = ((b * HKV + hh) * SP + split) * M + ro
        gl.store(pacc_ptr + part[:, None] * D + co[None, :], acc.to(pacc_ptr.dtype.element_ty))
        rs = gl.arange(0, M, layout=R_L)
        parts = ((b * HKV + hh) * SP + split) * M + rs
        gl.store(pml_ptr + parts * 2, m_i)
        gl.store(pml_ptr + parts * 2 + 1, l_i)

    @gluon.jit
    def _gw_merge_rows(qkv_ptr, b, h, split, HQ: gl.constexpr, HKV: gl.constexpr, G: gl.constexpr,
                       L: gl.constexpr, M: gl.constexpr, D: gl.constexpr, RS: gl.constexpr, R_PAD: gl.constexpr):
        """Program `split`'s slice of the merge: rows [split * RS, +RS) of the pair's 2 * M rows (KV heads h,
        h + 1), and their gate logits (loaded before the wait for the other programs)."""
        ML: gl.constexpr = gl.BlockedLayout([1, 4], [2, 16], [4, 1], [1, 0])
        r2 = split * RS + gl.arange(0, R_PAD, layout=gl.SliceLayout(1, ML))
        ok = (r2 < split * RS + RS) & (r2 < 2 * M)
        r2 = gl.minimum(r2, 2 * M - 1)
        hh = h + r2 // M
        r = r2 % M
        c = gl.arange(0, D, layout=gl.SliceLayout(0, ML))
        QKV_W: gl.constexpr = (2 * HQ + 2 * HKV) * D
        tok = b * L + r // G
        col = (hh * G + r % G)[:, None] * D + c[None, :]
        gate = gl.load(qkv_ptr + tok[:, None] * QKV_W + (HQ + 2 * HKV) * D + col).to(gl.float32)
        return hh, r, c, ok, tok, col, gate

    @gluon.jit
    def _gw_merge(pacc_ptr, pml_ptr, out_ptr, vs, b, hh, r, c, ok, tok, col, gate, HQ: gl.constexpr,
                  HKV: gl.constexpr, S: gl.constexpr, M: gl.constexpr, D: gl.constexpr, R_PAD: gl.constexpr):
        """_combine inside _gw_verify for _gw_merge_rows' slice: merge all S + 1 partials, then the per-head
        RMSNorm and sigmoid(gate)."""
        ML: gl.constexpr = gl.BlockedLayout([1, 4], [2, 16], [4, 1], [1, 0])
        SP: gl.constexpr = S + 1
        base = (b * HKV + hh) * SP * M + r
        m_max = gl.full([R_PAD], -1.0e30, gl.float32, layout=gl.SliceLayout(1, ML))
        for s in gl.static_range(SP):
            m_max = gl.maximum(m_max, gl.load(pml_ptr + (base + s * M) * 2, cache_modifier=".cg"))
        denom = gl.zeros([R_PAD], gl.float32, layout=gl.SliceLayout(1, ML))
        acc = gl.zeros([R_PAD, D], gl.float32, layout=ML)
        for s in gl.static_range(SP):
            w = gl.exp2(gl.load(pml_ptr + (base + s * M) * 2, cache_modifier=".cg") - m_max)
            denom += w * gl.load(pml_ptr + (base + s * M) * 2 + 1, cache_modifier=".cg")
            acc += w[:, None] * gl.load(pacc_ptr + (base + s * M)[:, None] * D + c[None, :],
                                        cache_modifier=".cg").to(gl.float32)
        o = acc / denom[:, None] * vs
        o = o * gl.rsqrt(gl.sum(o * o, 1) / D + 1e-5)[:, None]
        o = o / (1.0 + gl.exp2(gate * -1.4426950408889634))
        gl.store(out_ptr + tok[:, None] * (HQ * D) + col, o.to(gl.bfloat16), mask=ok[:, None])

    @gluon.jit
    def _gw_consumer(q, k_ring, v_ring, vh, ready, empty, lo, hi, ntiles, scale, pacc_ptr, pml_ptr, b, hh,
                     split, qkv_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, loc_ptr,
                     HEAD: gl.constexpr, HQ: gl.constexpr, HKV: gl.constexpr, G: gl.constexpr, L: gl.constexpr,
                     S: gl.constexpr, M: gl.constexpr, D: gl.constexpr, BN: gl.constexpr, NS: gl.constexpr):
        VL: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
        S_L: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1],
                                                      instr_shape=[16, BN, 32])
        O_L: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1],
                                                      instr_shape=[16, D, 16])
        P_L: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=O_L, k_width=2)
        m_i = gl.full([M], -1.0e30, gl.float32, layout=gl.SliceLayout(1, S_L))
        l_i = gl.zeros([M], gl.float32, layout=gl.SliceLayout(1, S_L))
        acc = gl.zeros([M, D], gl.float32, layout=O_L)
        n_s = gl.arange(0, BN, layout=gl.SliceLayout(0, S_L))
        for i in range(ntiles):
            s = i % NS
            mbarrier.wait(ready.index(s), (i // NS) & 1)
            s_tok = warpgroup_mma(q, k_ring.index(s).slice(HEAD * D, D, 1).permute((1, 0)),
                                  gl.zeros([M, BN], gl.float32, S_L), use_acc=False, is_async=True)
            vh.store(v_ring.index(s).slice(HEAD * D, D, 1).load(VL).to(gl.float16))
            fence_async_shared()
            t = lo + i * BN + n_s
            sc = warpgroup_mma_wait(0, deps=[s_tok])
            gl.barrier()
            mbarrier.arrive(empty.index(s))
            sc = gl.where((t < hi)[None, :], sc, float("-inf"))
            m_new = gl.maximum(m_i, gl.max(sc, 1) * scale)
            alpha = gl.exp2(m_i - m_new)
            p = gl.exp2(gl.fma(sc, scale, -m_new[:, None]))
            l_i = l_i * alpha + gl.sum(p, 1)
            m_i = m_new
            acc = acc * gl.convert_layout(alpha, gl.SliceLayout(1, O_L))[:, None]
            p16 = gl.convert_layout(p.to(gl.float16), P_L)
            a_tok = warpgroup_mma(p16, vh, acc, is_async=True)
            acc, p16 = warpgroup_mma_wait(0, deps=[a_tok, p16])
        _gw_store(pacc_ptr, pml_ptr, b, hh, split, m_i, l_i, acc, HKV, S + 1, M, D, gl.SliceLayout(1, S_L), O_L)

    @gluon.jit
    def _gw_producer_cv(pt_row, kbase, vbase, k_ring, v_ring, vh0, vh1, ready, empty, lo, hi, ntiles,
                        PAGE: gl.constexpr, ROW_W: gl.constexpr, BN: gl.constexpr, C: gl.constexpr,
                        D: gl.constexpr, NS: gl.constexpr, PW: gl.constexpr, LAG: gl.constexpr):
        """_gw_producer that also converts V to FP16 (per-head halves), LAG tiles behind its gathers."""
        GLY: gl.constexpr = gl.BlockedLayout([1, 16], [4, 8], [PW, 1], [1, 0])
        VL: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [PW, 1], [1, 0])
        n_g = gl.arange(0, BN, layout=gl.SliceLayout(1, GLY))
        c_g = gl.arange(0, C, layout=gl.SliceLayout(0, GLY))
        t = gl.maximum(gl.minimum(lo + n_g, hi - 1), 0)
        pg = gl.load(pt_row + t // PAGE)
        for i in range(ntiles + LAG):
            if i < ntiles:
                s = i % NS
                off = ((pg.to(gl.int64) * PAGE + t % PAGE) * ROW_W)[:, None] + c_g[None, :]
                t = gl.maximum(gl.minimum(lo + (i + 1) * BN + n_g, hi - 1), 0)
                pg = gl.load(pt_row + t // PAGE)
                mbarrier.wait(empty.index(s), ((i // NS) & 1) ^ 1)
                cp.async_copy_global_to_shared(k_ring.index(s), kbase + off)
                cp.async_copy_global_to_shared(v_ring.index(s), vbase + off)
            cp.commit_group()
            if i >= LAG:
                sj = (i - LAG) % NS
                cp.wait_group(LAG)
                gl.barrier()
                vt = v_ring.index(sj)
                vh0.index(sj).store(vt.slice(0, D, 1).load(VL).to(gl.float16))
                vh1.index(sj).store(vt.slice(D, D, 1).load(VL).to(gl.float16))
                fence_async_shared()
                # (mbarrier.arrive syncs the partition before its elected arrive.)
                mbarrier.arrive(ready.index(sj))

    @gluon.jit
    def _gw_consumer_cv(q, k_ring, vh, ready, empty, lo, hi, ntiles, scale, pacc_ptr, pml_ptr, b, hh, split,
                        HEAD: gl.constexpr, HKV: gl.constexpr, S: gl.constexpr, M: gl.constexpr, D: gl.constexpr,
                        BN: gl.constexpr, NS: gl.constexpr, PEEL: gl.constexpr, LSUM: gl.constexpr = False):
        S_L: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1],
                                                      instr_shape=[16, BN, 32])
        O_L: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1],
                                                      instr_shape=[16, D, 16])
        P_L: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=O_L, k_width=2)
        m_i = gl.full([M], -1.0e30, gl.float32, layout=gl.SliceLayout(1, S_L))
        l_i = gl.zeros([M], gl.float32, layout=gl.SliceLayout(1, S_L))
        acc = gl.zeros([M, D], gl.float32, layout=O_L)
        n_s = gl.arange(0, BN, layout=gl.SliceLayout(0, S_L))
        if LSUM:
            # Row sums kept per thread ([M, 8]: the 8 columns of each 8-wide group a quad holds), reduced
            # across the quad once at the end, as FA3 does.
            l8 = gl.sum(gl.reshape(gl.zeros([M, BN], gl.float32, S_L), [M, BN // 8, 8]), 1)
        for i in range(ntiles):
            s = i % NS
            mbarrier.wait(ready.index(s), (i // NS) & 1)
            s_tok = warpgroup_mma(q, k_ring.index(s).slice(HEAD * D, D, 1).permute((1, 0)),
                                  gl.zeros([M, BN], gl.float32, S_L), use_acc=False, is_async=True)
            sc = warpgroup_mma_wait(0, deps=[s_tok])
            if PEEL:
                # Only a chunk's last tile can run past its keys.
                if lo + (i + 1) * BN > hi:
                    sc = gl.where((lo + i * BN + n_s < hi)[None, :], sc, float("-inf"))
            else:
                sc = gl.where((lo + i * BN + n_s < hi)[None, :], sc, float("-inf"))
            m_new = gl.maximum(m_i, gl.max(sc, 1) * scale)
            alpha = gl.exp2(m_i - m_new)
            p = gl.exp2(gl.fma(sc, scale, -m_new[:, None]))
            if LSUM:
                l8 = l8 * gl.convert_layout(alpha, gl.SliceLayout(1, l8.type.layout))[:, None] + \
                    gl.sum(gl.reshape(p, [M, BN // 8, 8]), 1)
            else:
                l_i = l_i * alpha + gl.sum(p, 1)
            m_i = m_new
            acc = acc * gl.convert_layout(alpha, gl.SliceLayout(1, O_L))[:, None]
            p16 = gl.convert_layout(p.to(gl.float16), P_L)
            a_tok = warpgroup_mma(p16, vh.index(s), acc, is_async=True)
            acc, p16 = warpgroup_mma_wait(0, deps=[a_tok, p16])
            mbarrier.arrive(empty.index(s))
        if LSUM:
            l_i = gl.convert_layout(gl.sum(l8, 1), gl.SliceLayout(1, S_L))
        _gw_store(pacc_ptr, pml_ptr, b, hh, split, m_i, l_i, acc, HKV, S + 1, M, D, gl.SliceLayout(1, S_L), O_L)

    # ---- Balanced splits (plan gbal): the batch's key tiles cut into equal ranges, one per program ----
    # Program p (of P per KV head pair) takes tiles [p * T // P, (p + 1) * T // P) of the requests' tiles laid
    # end to end, so a range can span requests: one segment per request it touches, each stored at partial
    # id p + b (unique: moving to the next segment bumps p or b). Ids P + B + b hold the new tokens.
    @gluon.jit
    def _gw_bal(seq_ptr, p, B: gl.constexpr, B_PAD: gl.constexpr, P: gl.constexpr, BN: gl.constexpr,
                NW: gl.constexpr):
        # All B_PAD lengths in each warp (replicated across warps): the scan and sums stay inside a warp.
        LY: gl.constexpr = gl.BlockedLayout([max(1, B_PAD // 32)], [32], [NW], [0])
        idx = gl.arange(0, B_PAD, layout=LY)
        seqv = gl.load(seq_ptr + idx, mask=idx < B, other=0).to(gl.int32)
        tiles = (seqv + (BN - 1)) // BN
        incl = gl.associative_scan(tiles, 0, _gw_add)
        tt = gl.maximum(gl.sum(tiles, 0), 1)
        start = p * tt // P
        end = (p + 1) * tt // P
        b_first = gl.sum((incl <= start).to(gl.int32), 0)
        b_end = gl.sum(((incl - tiles < end) & (idx < B)).to(gl.int32), 0)
        nseg = gl.where(end > start, b_end - b_first, 0)
        return idx, seqv, tiles, start, end, b_first, nseg

    @gluon.jit
    def _gw_add(a, b):
        return a + b

    @gluon.jit
    def _gw_seg(idx, seqv, tiles, start, end, b, BN: gl.constexpr):
        """Segment of request b in tile range [start, end): its keys [lo, hi) and tile count."""
        excl = gl.sum(gl.where(idx < b, tiles, 0), 0)
        nb = gl.sum(gl.where(idx == b, tiles, 0), 0)
        sb = gl.sum(gl.where(idx == b, seqv, 0), 0)
        t0 = gl.maximum(start, excl) - excl
        t1 = gl.minimum(end, excl + nb) - excl
        lo = t0 * BN
        hi = gl.minimum(t1 * BN, sb)
        return lo, hi, gl.maximum(t1 - t0, 0)

    @gluon.jit
    def _gw_producer_bal(seq_ptr, pt_ptr, kbase, vbase, k_ring, v_ring, vh0, vh1, ready, empty, p,
                         B: gl.constexpr, B_PAD: gl.constexpr, P: gl.constexpr, MAX_PAGES: gl.constexpr,
                         PAGE: gl.constexpr, ROW_W: gl.constexpr, BN: gl.constexpr, C: gl.constexpr,
                         D: gl.constexpr, NS: gl.constexpr, PW: gl.constexpr, LAG: gl.constexpr):
        """_gw_producer_cv over the program's segments (the ring position carries across them)."""
        GLY: gl.constexpr = gl.BlockedLayout([1, 16], [4, 8], [PW, 1], [1, 0])
        VL: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [PW, 1], [1, 0])
        n_g = gl.arange(0, BN, layout=gl.SliceLayout(1, GLY))
        c_g = gl.arange(0, C, layout=gl.SliceLayout(0, GLY))
        idx, seqv, tiles, start, end, b_first, nseg = _gw_bal(seq_ptr, p, B, B_PAD, P, BN, PW)
        it = 0
        for sg in range(nseg):
            b = b_first + sg
            lo, hi, ntiles = _gw_seg(idx, seqv, tiles, start, end, b, BN)
            pt_row = pt_ptr + b.to(gl.int64) * MAX_PAGES
            t = gl.maximum(gl.minimum(lo + n_g, hi - 1), 0)
            pg = gl.load(pt_row + t // PAGE)
            for i in range(ntiles + LAG):
                if i < ntiles:
                    s = (it + i) % NS
                    off = ((pg.to(gl.int64) * PAGE + t % PAGE) * ROW_W)[:, None] + c_g[None, :]
                    t = gl.maximum(gl.minimum(lo + (i + 1) * BN + n_g, hi - 1), 0)
                    pg = gl.load(pt_row + t // PAGE)
                    mbarrier.wait(empty.index(s), (((it + i) // NS) & 1) ^ 1)
                    cp.async_copy_global_to_shared(k_ring.index(s), kbase + off)
                    cp.async_copy_global_to_shared(v_ring.index(s), vbase + off)
                cp.commit_group()
                if i >= LAG:
                    if i - LAG < ntiles:
                        sj = (it + i - LAG) % NS
                        cp.wait_group(LAG)
                        gl.barrier()
                        vt = v_ring.index(sj)
                        vh0.index(sj).store(vt.slice(0, D, 1).load(VL).to(gl.float16))
                        vh1.index(sj).store(vt.slice(D, D, 1).load(VL).to(gl.float16))
                        fence_async_shared()
                        mbarrier.arrive(ready.index(sj))
            it += ntiles

    @gluon.jit
    def _gw_consumer_bal(qkv_ptr, q_s, k_ring, vh, ready, empty, seq_ptr, pid, h, scale, pacc_ptr, pml_ptr,
                         HEAD: gl.constexpr, HQ: gl.constexpr, HKV: gl.constexpr, G: gl.constexpr, L: gl.constexpr,
                         B: gl.constexpr, B_PAD: gl.constexpr, P: gl.constexpr, NID: gl.constexpr, M: gl.constexpr,
                         D: gl.constexpr, BN: gl.constexpr, NS: gl.constexpr):
        """_gw_consumer_cv (PEEL, LSUM) over the program's segments: each loads its request's queries and stores
        its partial at id p + b."""
        S_L: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1],
                                                      instr_shape=[16, BN, 32])
        O_L: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1],
                                                      instr_shape=[16, D, 16])
        P_L: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=O_L, k_width=2)
        QL: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
        QKV_W: gl.constexpr = (2 * HQ + 2 * HKV) * D
        r = gl.arange(0, M, layout=gl.SliceLayout(1, QL))
        cq = gl.arange(0, D, layout=gl.SliceLayout(0, QL))
        n_s = gl.arange(0, BN, layout=gl.SliceLayout(0, S_L))
        idx, seqv, tiles, start, end, b_first, nseg = _gw_bal(seq_ptr, pid, B, B_PAD, P, BN, 4)
        it = 0
        for sg in range(nseg):
            b = b_first + sg
            lo, hi, ntiles = _gw_seg(idx, seqv, tiles, start, end, b, BN)
            x = gl.load(qkv_ptr + (b * L + r // G)[:, None] * QKV_W + ((h + HEAD) * G + r % G)[:, None] * D
                        + cq[None, :]).to(gl.float32)
            q_s.store((x * gl.rsqrt(gl.sum(x * x, 1) / D + 1e-5)[:, None]).to(gl.float8e4nv))
            fence_async_shared()
            gl.barrier()
            m_i = gl.full([M], -1.0e30, gl.float32, layout=gl.SliceLayout(1, S_L))
            acc = gl.zeros([M, D], gl.float32, layout=O_L)
            l8 = gl.sum(gl.reshape(gl.zeros([M, BN], gl.float32, S_L), [M, BN // 8, 8]), 1)
            for i in range(ntiles):
                s = (it + i) % NS
                mbarrier.wait(ready.index(s), ((it + i) // NS) & 1)
                s_tok = warpgroup_mma(q_s, k_ring.index(s).slice(HEAD * D, D, 1).permute((1, 0)),
                                      gl.zeros([M, BN], gl.float32, S_L), use_acc=False, is_async=True)
                sc = warpgroup_mma_wait(0, deps=[s_tok])
                if lo + (i + 1) * BN > hi:
                    sc = gl.where((lo + i * BN + n_s < hi)[None, :], sc, float("-inf"))
                m_new = gl.maximum(m_i, gl.max(sc, 1) * scale)
                alpha = gl.exp2(m_i - m_new)
                p = gl.exp2(gl.fma(sc, scale, -m_new[:, None]))
                l8 = l8 * gl.convert_layout(alpha, gl.SliceLayout(1, l8.type.layout))[:, None] + \
                    gl.sum(gl.reshape(p, [M, BN // 8, 8]), 1)
                m_i = m_new
                acc = acc * gl.convert_layout(alpha, gl.SliceLayout(1, O_L))[:, None]
                p16 = gl.convert_layout(p.to(gl.float16), P_L)
                a_tok = warpgroup_mma(p16, vh.index(s), acc, is_async=True)
                acc, p16 = warpgroup_mma_wait(0, deps=[a_tok, p16])
                mbarrier.arrive(empty.index(s))
            it += ntiles
            l_i = gl.convert_layout(gl.sum(l8, 1), gl.SliceLayout(1, S_L))
            _gw_store(pacc_ptr, pml_ptr, 0, h + HEAD, pid + b, m_i, l_i, acc, HKV, NID, M, D, gl.SliceLayout(1, S_L),
                      O_L)

    # ---- Decode steps (q_len 1) on full layers at batch * hkv > 2 (Gluon, one warp per program) ----
    # Like _attn_split's hpp-2 programs (one split of one request's keys for a KV head pair, block-diagonal
    # queries [2G, 2D] against whole 128-byte rows), but QK^T runs as FP8 mma.sync (Triton's MMAv2 path upcasts
    # every K byte to FP16 on sm_90), V is read from shared memory 16 bytes at a time, and the pair's new token is
    # written and attended by the last split into its own partial slot S (no separate launch). Same bytes, ~15-20%
    # less work per key (H100, all reads from L2: decode 16/32 full 84/157 -> 69/132 us), which pays where
    # per-SM bandwidth is higher (tools/emu.py: 137/261 -> 132/249 us; H100 itself: a tie).
    @gluon.jit
    def _gd_full(qkv_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, pt_ptr, seq_ptr, loc_ptr, pacc_ptr, pml_ptr, qk_scale,
                 cs_ptr, HQ: gl.constexpr, HKV: gl.constexpr, G: gl.constexpr, D: gl.constexpr, PAGE: gl.constexpr,
                 MAX_PAGES: gl.constexpr, S: gl.constexpr, BN: gl.constexpr, NS: gl.constexpr,
                 TPV: gl.constexpr = False, WINDOW: gl.constexpr = 0, ROPE: gl.constexpr = False,
                 SEP: gl.constexpr = False):
        split = gl.program_id(0)
        b = gl.program_id(1)
        h = gl.program_id(2) * 2
        # Let the merge kernel launch now (programmatic dependent launch); it waits for this grid.
        gl.inline_asm_elementwise("griddepcontrol.launch_dependents;", "=r", [], dtype=gl.int32, is_pure=False,
                                  pack=1)
        C: gl.constexpr = 2 * D
        M: gl.constexpr = 2 * G
        QKV_W: gl.constexpr = (2 * HQ + 2 * HKV) * D
        ROW_W: gl.constexpr = HKV * D
        MMA: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[1, 1], instr_shape=[16, 8])
        QA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=MMA, k_width=4)
        KB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=MMA, k_width=4)
        PA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=MMA, k_width=2)
        VB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=MMA, k_width=2)
        VA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=MMA, k_width=2)
        PB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=MMA, k_width=2)
        # 16 bytes per thread. (Converting FP8 queries into QA from a [1, 4]-per-thread layout gave wrong values.)
        BL: gl.constexpr = gl.BlockedLayout([1, 16], [4, 8], [1, 1], [1, 0])
        SH: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 8, [1, 0])

        seq = gl.load(seq_ptr + b)
        # Each request's cached keys (sliding layers: the last WINDOW - 1) in S equal tile-aligned chunks.
        if WINDOW > 0:
            base = gl.maximum(seq - (WINDOW - 1), 0)
        else:
            base = seq * 0
        chunk = gl.cdiv(gl.cdiv(seq - base, S), BN) * BN
        lo = base + split * chunk
        hi = gl.minimum(lo + chunk, seq)
        ntiles = gl.cdiv(gl.maximum(hi - lo, 0), BN)
        pt_row = pt_ptr + b.to(gl.int64) * MAX_PAGES
        kbase = kc_ptr + h * D
        vbase = vc_ptr + h * D
        k_ring = gl.allocate_shared_memory(gl.float8e4nv, [NS, BN, C], SH)
        v_ring = gl.allocate_shared_memory(gl.float8e4nv, [NS, BN, C], SH)
        n_g = gl.arange(0, BN, layout=gl.SliceLayout(1, BL))
        c_g = gl.arange(0, C, layout=gl.SliceLayout(0, BL))
        all_c = c_g[None, :] >= 0
        # The first NS - 1 tiles in flight (cp.async zero-fills rows past hi instead of reading them).
        for j in gl.static_range(NS - 1):
            t = lo + j * BN + n_g
            ok = t < hi
            pg = gl.load(pt_row + t // PAGE, mask=ok, other=0)
            off = ((pg.to(gl.int64) * PAGE + t % PAGE) * ROW_W)[:, None] + c_g[None, :]
            cp.async_copy_global_to_shared(k_ring.index(j), kbase + off, mask=ok[:, None] & all_c)
            cp.async_copy_global_to_shared(v_ring.index(j), vbase + off, mask=ok[:, None] & all_c)
            cp.commit_group()
        t = lo + (NS - 1) * BN + n_g
        pg = gl.load(pt_row + t // PAGE, mask=t < hi, other=0)

        # Block-diagonal queries: row r is query head (h + r // G) * G + r % G, nonzero only in its own KV head's
        # D columns. RMSNorm, then E4M3.
        r = gl.arange(0, M, layout=gl.SliceLayout(1, BL))
        hp = r // G
        q_ok = (c_g // D)[None, :] == hp[:, None]
        x = gl.load(qkv_ptr + b * QKV_W + ((h + hp) * G + r % G)[:, None] * D + (c_g % D)[None, :], mask=q_ok,
                    other=0.0).to(gl.float32)
        x = gl.where(q_ok, x, 0.0)
        rs = gl.rsqrt(gl.sum(x * x, 1) / D + 1e-5)[:, None]
        if ROPE:
            # Interleaved RoPE at position seq (the norm commutes with the rotation).
            xp = gl.load(qkv_ptr + b * QKV_W + ((h + hp) * G + r % G)[:, None] * D + ((c_g % D) ^ 1)[None, :],
                         mask=q_ok, other=0.0).to(gl.float32)
            cs_off = seq * D + (c_g % D) // 2
            cos = gl.load(cs_ptr + cs_off)[None, :]
            sin = (gl.load(cs_ptr + cs_off + D // 2) * gl.where(c_g % 2 == 0, -1.0, 1.0))[None, :]
            x = x * cos + xp * sin
        x8 = (x * rs).to(gl.float8e4nv)
        q = gl.convert_layout(x8, QA)
        k_scale = gl.load(ks_ptr)
        scale = qk_scale * k_scale

        m_i = gl.full([M], -1.0e30, gl.float32, layout=gl.SliceLayout(1, MMA))
        l_i = gl.zeros([M], gl.float32, layout=gl.SliceLayout(1, MMA))
        if TPV:
            # Transposed P@V per head: O^T [D, G] += V^T [D, BN] @ P^T [BN, G] (only the diagonal blocks: half the
            # MMAs and accumulator registers of the block-diagonal form). V staged once per tile as FP16.
            vh = gl.allocate_shared_memory(gl.float16, [BN, C], gl.SwizzledSharedLayout(8, 1, 8, [1, 0]))
            acc0 = gl.zeros([D, G], gl.float32, layout=MMA)
            acc1 = gl.zeros([D, G], gl.float32, layout=MMA)
        else:
            acc = gl.zeros([M, C], gl.float32, layout=MMA)
        n_s = gl.arange(0, BN, layout=gl.SliceLayout(0, MMA))
        for i in range(ntiles):
            sj = (i + NS - 1) % NS
            ok = t < hi
            off = ((pg.to(gl.int64) * PAGE + t % PAGE) * ROW_W)[:, None] + c_g[None, :]
            gl.barrier()  # slot sj was read last iteration
            cp.async_copy_global_to_shared(k_ring.index(sj), kbase + off, mask=ok[:, None] & all_c)
            cp.async_copy_global_to_shared(v_ring.index(sj), vbase + off, mask=ok[:, None] & all_c)
            cp.commit_group()
            t = t + BN
            pg = gl.load(pt_row + t // PAGE, mask=t < hi, other=0)
            cp.wait_group(NS - 1)
            gl.barrier()
            s = i % NS
            sc = mma_v2(q, k_ring.index(s).permute((1, 0)).load(KB), gl.zeros([M, BN], gl.float32, MMA))
            if lo + (i + 1) * BN > hi:
                sc = gl.where((lo + i * BN + n_s < hi)[None, :], sc, float("-inf"))
            m_new = gl.maximum(m_i, gl.max(sc, 1) * scale)
            alpha = gl.exp2(m_i - m_new)
            p = gl.exp2(gl.fma(sc, scale, -m_new[:, None]))
            l_i = l_i * alpha + gl.sum(p, 1)
            m_i = m_new
            if TPV:
                a0, a1 = gl.split(gl.permute(gl.reshape(alpha, [2, G]), (1, 0)))
                acc0 = acc0 * gl.convert_layout(a0, gl.SliceLayout(0, MMA))[None, :]
                acc1 = acc1 * gl.convert_layout(a1, gl.SliceLayout(0, MMA))[None, :]
                p0, p1 = gl.split(gl.permute(gl.reshape(p.to(gl.float16), [2, G, BN]), (2, 1, 0)))
                vh.store(v_ring.index(s).load(BL).to(gl.float16))
                gl.barrier()
                acc0 = mma_v2(vh.slice(0, D, 1).permute((1, 0)).load(VA), gl.convert_layout(p0, PB), acc0)
                acc1 = mma_v2(vh.slice(D, D, 1).permute((1, 0)).load(VA), gl.convert_layout(p1, PB), acc1)
            else:
                acc = acc * alpha[:, None]
                v = gl.convert_layout(v_ring.index(s).load(BL).to(gl.float16), VB)
                acc = mma_v2(gl.convert_layout(p.to(gl.float16), PA), v, acc)
        cp.wait_group(0)

        # Partials [B, HKV, S + 1, G, D]: each row's own head's D columns of acc.
        ro = gl.arange(0, M, layout=gl.SliceLayout(1, MMA))
        co = gl.arange(0, C, layout=gl.SliceLayout(0, MMA))
        part = ((b * HKV + h + ro // G) * (S + 1) + split) * G + ro % G
        own = (co // D)[None, :] == (ro // G)[:, None]
        # SEP: an extra program (split S) only handles the new token; it stores no key partial.
        keyp = split < S
        own = own & keyp
        if TPV:
            dd = gl.arange(0, D, layout=gl.SliceLayout(1, MMA))
            gg = gl.arange(0, G, layout=gl.SliceLayout(0, MMA))
            pt0 = ((b * HKV + h) * (S + 1) + split) * G + gg
            gl.store(pacc_ptr + pt0[None, :] * D + dd[:, None], acc0.to(pacc_ptr.dtype.element_ty))
            gl.store(pacc_ptr + (pt0 + (S + 1) * G)[None, :] * D + dd[:, None], acc1.to(pacc_ptr.dtype.element_ty))
        else:
            gl.store(pacc_ptr + part[:, None] * D + (co % D)[None, :], acc.to(pacc_ptr.dtype.element_ty), mask=own)
        gl.store(pml_ptr + part * 2, m_i, mask=keyp)
        gl.store(pml_ptr + part * 2 + 1, l_i, mask=keyp)
        if split == (S if SEP else S - 1):
            # The step's new token for both KV heads: normalize K, store K and V as FP8 at cache_loc (row 0 of
            # each head's block), and write its one-key partial (max = its score, sum = 1, acc = V) to slot S.
            loc = gl.load(loc_ptr + b)
            zero = gl.zeros([M, C], gl.float32, BL)
            kx = gl.where(q_ok, gl.load(qkv_ptr + b * QKV_W + (HQ + h) * D + c_g)[None, :].to(gl.float32) + zero, 0.0)
            krs = gl.rsqrt(gl.sum(kx * kx, 1) / D + 1e-5)[:, None]
            if ROPE:
                kxp = gl.where(q_ok, gl.load(qkv_ptr + b * QKV_W + (HQ + h) * D + (c_g ^ 1))[None, :].to(gl.float32)
                               + zero, 0.0)
                kx = kx * cos + kxp * sin
            kq = (kx * krs / k_scale).to(gl.float8e4nv)
            vx = gl.load(qkv_ptr + b * QKV_W + (HQ + HKV + h) * D + c_g)[None, :].to(gl.float32) + zero
            vq = (gl.where(q_ok, vx, 0.0) / gl.load(vs_ptr)).to(gl.float8e4nv)
            first = q_ok & ((r % G) == 0)[:, None]
            dst = (loc.to(gl.int64) * ROW_W + h * D + c_g)[None, :] + gl.zeros([M, C], gl.int64, BL)
            gl.store(kc_ptr + dst, kq, mask=first)
            gl.store(vc_ptr + dst, vq, mask=first)
            sn = gl.sum(x8.to(gl.float32) * kq.to(gl.float32), 1) * scale
            pn = ((b * HKV + h + hp) * (S + 1) + S) * G + r % G
            gl.store(pacc_ptr + pn[:, None] * D + (c_g % D)[None, :], vq.to(gl.float32).to(pacc_ptr.dtype.element_ty),
                     mask=q_ok)
            gl.store(pml_ptr + pn * 2, sn)
            gl.store(pml_ptr + pn * 2 + 1, gl.full([M], 1.0, gl.float32, gl.SliceLayout(1, BL)))

    @gluon.jit
    def _gw_verify(qkv_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, pt_ptr, seq_ptr, loc_ptr, pacc_ptr, pml_ptr, qk_scale,
                   HQ: gl.constexpr, HKV: gl.constexpr, G: gl.constexpr, L: gl.constexpr, D: gl.constexpr,
                   PAGE: gl.constexpr, MAX_PAGES: gl.constexpr, S: gl.constexpr, BN: gl.constexpr,
                   NS: gl.constexpr, PW: gl.constexpr, PREGS: gl.constexpr, CREGS: gl.constexpr,
                   LAG: gl.constexpr = 0, PEEL: gl.constexpr = False, LSUM: gl.constexpr = False,
                   FUSE: gl.constexpr = False, cnt_ptr=None, out_ptr=None, RS: gl.constexpr = 1,
                   R_PAD: gl.constexpr = 1, BAL: gl.constexpr = False, B: gl.constexpr = 1,
                   B_PAD: gl.constexpr = 1, NID: gl.constexpr = 1, NP: gl.constexpr = 1):
        M: gl.constexpr = L * G
        C: gl.constexpr = 2 * D
        split = gl.program_id(0)
        b = gl.program_id(1)
        h = gl.program_id(2) * 2
        # Let the merge kernel launch now (programmatic dependent launch); it waits for this grid.
        gl.inline_asm_elementwise("griddepcontrol.launch_dependents;", "=r", [], dtype=gl.int32, is_pure=False, pack=1)
        QKV_W: gl.constexpr = (2 * HQ + 2 * HKV) * D
        ROW_W: gl.constexpr = HKV * D
        QL: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
        SH8: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=8, rank=2)
        SH16: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
        VSH: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
        SQ: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=64, element_bitwidth=8, rank=2)

        seq = gl.load(seq_ptr + b)
        # Each request's cached keys in S equal tile-aligned chunks.
        chunk = gl.cdiv(gl.cdiv(seq, S), BN) * BN
        lo = split * chunk
        hi = gl.minimum(lo + chunk, seq)
        ntiles = gl.cdiv(gl.maximum(hi - lo, 0), BN)
        pt_row = pt_ptr + b.to(gl.int64) * MAX_PAGES

        k_ring = gl.allocate_shared_memory(gl.float8e4nv, [NS, BN, C], SH8)
        v_ring = gl.allocate_shared_memory(gl.float8e4nv, [NS, BN, C], VSH)
        if LAG:
            vh0 = gl.allocate_shared_memory(gl.float16, [NS, BN, D], SH16)
            vh1 = gl.allocate_shared_memory(gl.float16, [NS, BN, D], SH16)
        else:
            vh0 = gl.allocate_shared_memory(gl.float16, [BN, D], SH16)
            vh1 = gl.allocate_shared_memory(gl.float16, [BN, D], SH16)
        ready = gl.allocate_shared_memory(gl.int64, [NS, 1], mbarrier.MBarrierLayout())
        empty = gl.allocate_shared_memory(gl.int64, [NS, 1], mbarrier.MBarrierLayout())
        for j in gl.static_range(NS):
            mbarrier.init(ready.index(j), count=1 if LAG else PW * 32)
            mbarrier.init(empty.index(j), count=2)
        r = gl.arange(0, M, layout=gl.SliceLayout(1, QL))
        cq = gl.arange(0, D, layout=gl.SliceLayout(0, QL))
        q_off = (b * L + r // G)[:, None] * QKV_W + (h * G + r % G)[:, None] * D + cq[None, :]
        if BAL:
            # (Each consumer loads its segments' queries.)
            q0 = gl.allocate_shared_memory(gl.float8e4nv, [M, D], SQ)
            q1 = gl.allocate_shared_memory(gl.float8e4nv, [M, D], SQ)
        else:
            x0 = gl.load(qkv_ptr + q_off).to(gl.float32)
            x0 = (x0 * gl.rsqrt(gl.sum(x0 * x0, 1) / D + 1e-5)[:, None]).to(gl.float8e4nv)
            q0 = gl.allocate_shared_memory(gl.float8e4nv, [M, D], SQ, x0)
            x1 = gl.load(qkv_ptr + q_off + G * D).to(gl.float32)
            x1 = (x1 * gl.rsqrt(gl.sum(x1 * x1, 1) / D + 1e-5)[:, None]).to(gl.float8e4nv)
            q1 = gl.allocate_shared_memory(gl.float8e4nv, [M, D], SQ, x1)
        scale = qk_scale * gl.load(ks_ptr)
        fence_async_shared()
        gl.barrier()
        if BAL:
            gl.warp_specialize([
                (_gw_consumer_bal, (qkv_ptr, q0, k_ring, vh0, ready, empty, seq_ptr, split, h, scale, pacc_ptr,
                                    pml_ptr, 0, HQ, HKV, G, L, B, B_PAD, NP, NID, M, D, BN, NS)),
                (_gw_consumer_bal, (qkv_ptr, q1, k_ring, vh1, ready, empty, seq_ptr, split, h, scale, pacc_ptr,
                                    pml_ptr, 1, HQ, HKV, G, L, B, B_PAD, NP, NID, M, D, BN, NS)),
                (_gw_producer_bal, (seq_ptr, pt_ptr, kc_ptr + h * D, vc_ptr + h * D, k_ring, v_ring, vh0, vh1, ready,
                                    empty, split, B, B_PAD, NP, MAX_PAGES, PAGE, ROW_W, BN, C, D, NS, PW,
                                    LAG)),
            ], [4, PW], [CREGS, PREGS])
        elif LAG:
            gl.warp_specialize([
                (_gw_consumer_cv, (q0, k_ring, vh0, ready, empty, lo, hi, ntiles, scale, pacc_ptr, pml_ptr, b, h,
                                   split, 0, HKV, S, M, D, BN, NS, PEEL, LSUM)),
                (_gw_consumer_cv, (q1, k_ring, vh1, ready, empty, lo, hi, ntiles, scale, pacc_ptr, pml_ptr, b, h + 1,
                                   split, 1, HKV, S, M, D, BN, NS, PEEL, LSUM)),
                (_gw_producer_cv, (pt_row, kc_ptr + h * D, vc_ptr + h * D, k_ring, v_ring, vh0, vh1, ready, empty,
                                   lo, hi, ntiles, PAGE, ROW_W, BN, C, D, NS, PW, LAG)),
            ], [4, PW], [CREGS, PREGS])
        else:
          gl.warp_specialize([
            (_gw_consumer, (q0, k_ring, v_ring, vh0, ready, empty, lo, hi, ntiles, scale, pacc_ptr, pml_ptr, b, h,
                            split, qkv_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, loc_ptr, 0, HQ, HKV, G, L, S, M, D, BN,
                            NS)),
            (_gw_consumer, (q1, k_ring, v_ring, vh1, ready, empty, lo, hi, ntiles, scale, pacc_ptr, pml_ptr, b,
                            h + 1, split, qkv_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, loc_ptr, 1, HQ, HKV, G, L, S, M, D,
                            BN, NS)),
            (_gw_producer, (pt_row, kc_ptr + h * D, vc_ptr + h * D, k_ring, v_ring, ready, empty, lo, hi, ntiles,
                            PAGE, ROW_W, BN, C, NS, PW)),
        ], [4, PW], [CREGS, PREGS])
        if FUSE:
            # The merge, in this grid: once all S programs of this (request, KV head pair) have stored their
            # partials, each merges a slice of the rows. One counter word per pair: arrivals in the low 16
            # bits, a generation above; the last arrival clears the arrivals and bumps the generation, which
            # the others wait on (so each call leaves the arrivals at 0). The plan keeps this grid to one
            # wave, so every program waited for is resident. (Fences as cooperative groups' grid sync: the
            # partition barrier, then the acq_rel atomics.)
            hh, r, c, ok, tok, col, gate = _gw_merge_rows(qkv_ptr, b, h, split, HQ, HKV, G, L, M, D, RS, R_PAD)
            vs = gl.load(vs_ptr)
            cnt = cnt_ptr + b * (HKV // 2) + gl.program_id(2)
            gl.barrier()
            old = gl.atomic_add(cnt, 1, sem="acq_rel", scope="gpu")
            if (old & 0xFFFF) == S - 1:
                gl.atomic_add(cnt, 0x10000 - S, sem="relaxed", scope="gpu")
            else:
                while ((gl.load(cnt, volatile=True) ^ old) >> 16) == 0:
                    pass
            gl.inline_asm_elementwise("fence.acq_rel.gpu;", "=r", [], dtype=gl.int32, is_pure=False, pack=1)
        # This grid was launched alongside _new_tokens (which needs nothing from it); finish (or merge) only
        # after it, so the merge sees slot S too.
        gl.inline_asm_elementwise("griddepcontrol.wait;", "=r", [], dtype=gl.int32, is_pure=False, pack=1)
        if FUSE:
            _gw_merge(pacc_ptr, pml_ptr, out_ptr, vs, b, hh, r, c, ok, tok, col, gate, HQ, HKV, S, M, D, R_PAD)


@triton.jit
def _norm_rope(ptr, off, ok, c, pos, cs_ptr, ROPE: tl.constexpr, D: tl.constexpr, HPP: tl.constexpr):
    """[R, HPP*D] rows of HPP heads at ptr + off: per-head rmsnorm (no weight), then interleaved
    RoPE at pos [R]. Masked-out entries are 0."""
    x = tl.load(ptr + off, mask=ok, other=0.0).to(tl.float32)
    sq = x * x
    if HPP == 1:
        r = tl.rsqrt(tl.sum(sq, 1) / D + 1e-5)[:, None]
    else:
        first = (c < D)[None, :]
        r0 = tl.rsqrt(tl.sum(tl.where(first, sq, 0.0), 1) / D + 1e-5)
        r1 = tl.rsqrt(tl.sum(tl.where(first, 0.0, sq), 1) / D + 1e-5)
        r = tl.where(first, r0[:, None], r1[:, None])
    if ROPE:
        xp = tl.load(ptr + (off ^ 1), mask=ok, other=0.0).to(tl.float32)
        cs_off = pos[:, None] * D + ((c % D) // 2)[None, :]
        cos = tl.load(cs_ptr + cs_off, mask=ok, other=1.0)
        sin = tl.load(cs_ptr + cs_off + D // 2, mask=ok, other=0.0)
        sign = tl.where(c % 2 == 0, -1.0, 1.0)
        x = x * cos + sign[None, :] * xp * sin
    return x * r


@triton.jit
def _tile(t, hi, pt_row, kc_ptr, vc_ptr, q, pos, c, m_i, l_i, acc, scale, hp0, MASKED: tl.constexpr,
          PAGE: tl.constexpr, ROW_W: tl.constexpr, D: tl.constexpr, WINDOW: tl.constexpr, QK8: tl.constexpr,
          DOT: tl.constexpr):
    """One tile of cached keys at positions t through the online softmax. MASKED: some t >= hi."""
    if MASKED:
        # Past the end, gather the last valid row again (no load masks); the scores are masked.
        t_ok = t < hi
        t_ld = tl.minimum(t, hi - 1)
    else:
        t_ld = t
    page = tl.load(pt_row + t_ld // PAGE)
    row = (page.to(tl.int64) * PAGE + t_ld % PAGE) * ROW_W + hp0 * D
    k = tl.load(kc_ptr + row[:, None] + c[None, :])
    v = tl.load(vc_ptr + row[:, None] + c[None, :])
    s = tl.dot(q, tl.trans(k if QK8 else k.to(DOT)))
    if MASKED:
        ok = t_ok[None, :]
        if WINDOW > 0:
            ok = ok & (t[None, :] > pos[:, None] - WINDOW)
        s = tl.where(ok, s, float("-inf"))
    elif WINDOW > 0:
        s = tl.where(t[None, :] > pos[:, None] - WINDOW, s, float("-inf"))
    # The scale (> 0) goes into the row maxima and the exponent's FMA, not every score.
    m_new = tl.maximum(m_i, tl.max(s, 1) * scale)
    alpha = tl.exp2(m_i - m_new)
    p = tl.exp2(tl.fma(s, scale, -m_new[:, None]))
    l_i = l_i * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None] + tl.dot(p.to(DOT), v.to(DOT))
    return m_new, l_i, acc


@triton.jit
def _new(split, b, seq, j, q, m_i, l_i, acc, hp0, c, scale, k_scale, qkv_ptr, cs_ptr, kc_ptr,
         vc_ptr, vs_ptr, loc_ptr, HQ: tl.constexpr, HKV: tl.constexpr, HPP: tl.constexpr,
         L: tl.constexpr, L_PAD: tl.constexpr, MG: tl.constexpr, D: tl.constexpr, ROPE: tl.constexpr,
         S: tl.constexpr, QK8: tl.constexpr, DOT: tl.constexpr, SEP: tl.constexpr = False):
    """Split 0 (or with SEP the extra program S, which has no cached keys) attends the block's own tokens (and stores them in the cache). Done before the
    cached keys, so these loads overlap the page-table and cache trips instead of following them."""
    QKV_W: tl.constexpr = (2 * HQ + 2 * HKV) * D
    ROW_W: tl.constexpr = HKV * D
    if split == (S if SEP else 0):
        # The block's own tokens: normalize, store as FP8, attend causally.
        jn = tl.arange(0, L_PAD)
        jn_ok = jn < L
        tok = b * L + jn
        kn_off = (tok * QKV_W + (HQ + hp0) * D)[:, None] + c[None, :]
        kn = _norm_rope(qkv_ptr, kn_off, jn_ok[:, None], c, seq + jn, cs_ptr, ROPE, D, HPP)
        vn = tl.load(qkv_ptr + kn_off + HKV * D, mask=jn_ok[:, None], other=0.0)
        kq = (kn / k_scale).to(tl.float8e4nv)
        vq = (vn.to(tl.float32) / tl.load(vs_ptr)).to(tl.float8e4nv)
        slot = tl.load(loc_ptr + tok, mask=jn_ok, other=0) * ROW_W + hp0 * D
        tl.store(kc_ptr + slot[:, None] + c[None, :], kq, mask=jn_ok[:, None])
        tl.store(vc_ptr + slot[:, None] + c[None, :], vq, mask=jn_ok[:, None])
        s = tl.dot(q, tl.trans(kq if QK8 else kq.to(DOT))) * scale
        ok = jn_ok[None, :] & (jn[None, :] <= j[:, None])
        s = tl.where(ok, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(DOT), vq.to(DOT))
        m_i = m_new

    return m_i, l_i, acc


@triton.jit
def _finish(split, b, j, m_i, l_i, acc, hp0, hp, rr, r_ok, q_ok, c, pacc_ptr, pml_ptr, HKV: tl.constexpr,
            MG: tl.constexpr, D: tl.constexpr, S: tl.constexpr, SEP: tl.constexpr = False):
    """Every split writes its partial (max, sum, accumulator) for KV heads hp0 .. hp0 + HPP - 1."""
    # Partials [B, HKV, S, MG]: each row's own head's D columns of acc.
    part = ((b * HKV + hp0 + hp) * (S + SEP) + split) * MG + rr
    tl.store(pml_ptr + part * 2, m_i, mask=r_ok)
    tl.store(pml_ptr + part * 2 + 1, l_i, mask=r_ok)
    tl.store(pacc_ptr + part[:, None] * D + (c % D)[None, :], acc.to(pacc_ptr.dtype.element_ty), mask=q_ok)


@triton.jit
def _attn_split(qkv_ptr, cs_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, pt_ptr, seq_ptr, loc_ptr,
                pacc_ptr, pml_ptr, cnt_ptr, out_ptr, qk_scale,
                HQ: tl.constexpr, HKV: tl.constexpr, HPP: tl.constexpr, G: tl.constexpr, L: tl.constexpr,
                MG: tl.constexpr, M_PAD: tl.constexpr, L_PAD: tl.constexpr, D: tl.constexpr,
                PAGE: tl.constexpr, MAX_PAGES: tl.constexpr, WINDOW: tl.constexpr, ROPE: tl.constexpr,
                S: tl.constexpr, CHUNK: tl.constexpr, BLOCK_N: tl.constexpr, STAGES: tl.constexpr,
                FUSE: tl.constexpr, S_PAD: tl.constexpr, QK8: tl.constexpr, PDL: tl.constexpr,
                DOT: tl.constexpr, PEEL: tl.constexpr = False, DUAL: tl.constexpr = False,
                BALANCE: tl.constexpr = False, SEP: tl.constexpr = False):
    """One program: a split of request b's context for HPP adjacent KV heads, which share each
    cache row's contiguous HPP*D bytes. Queries are block-diagonal [HPP*MG, HPP*D] (row r is KV
    head hp = r // MG of the pair, and holds values only in that head's D columns), so one
    [rows, HPP*D] x [HPP*D, N] product scores every head against its own keys."""
    split = tl.program_id(0)
    b = tl.program_id(1)
    # DUAL (with HPP 1): the program also runs KV head hp0 + 1 as a second, independent chain.
    hp0 = tl.program_id(2) * (2 if DUAL else HPP)
    if PDL:
        # Let the merge kernel launch now; it waits on-device for this grid to finish.
        gdc_launch_dependents()
    QKV_W: tl.constexpr = (2 * HQ + 2 * HKV) * D
    ROW_W: tl.constexpr = HKV * D
    C: tl.constexpr = HPP * D

    seq = tl.load(seq_ptr + b)
    c = tl.arange(0, C)
    r = tl.arange(0, M_PAD)
    hp = r // MG
    rr = r % MG  # token rr // G, query head (hp0 + hp) * G + rr % G
    r_ok = r < HPP * MG
    j = rr // G
    pos = seq + j
    q_off = (b * L + j)[:, None] * QKV_W + ((hp0 + hp) * G + rr % G)[:, None] * D + (c % D)[None, :]
    q_ok = r_ok[:, None] & ((c // D)[None, :] == hp[:, None])
    q = _norm_rope(qkv_ptr, q_off, q_ok, c, pos, cs_ptr, ROPE, D, HPP)
    if QK8:
        # Queries rounded to E4M3 (as FA3 does with an FP8 cache): QK^T runs on the FP8 cache as is.
        q = q.to(tl.float8e4nv)
    else:
        q = q.to(DOT)
    q1 = q
    if DUAL:
        q1 = _norm_rope(qkv_ptr, q_off + G * D, q_ok, c, pos, cs_ptr, ROPE, D, HPP)
        q1 = q1.to(tl.float8e4nv) if QK8 else q1.to(DOT)
    k_scale = tl.load(ks_ptr)
    scale = qk_scale * k_scale  # log2 units, applied to q . k_fp8

    m_i = tl.full([M_PAD], -1.0e30, tl.float32)
    l_i = tl.zeros([M_PAD], tl.float32)
    acc = tl.zeros([M_PAD, C], tl.float32)
    m1, l1, acc1 = m_i, l_i, acc
    m_i, l_i, acc = _new(split, b, seq, j, q, m_i, l_i, acc, hp0, c, scale, k_scale, qkv_ptr, cs_ptr, kc_ptr,
                         vc_ptr, vs_ptr, loc_ptr, HQ, HKV, HPP, L, L_PAD, MG, D, ROPE, S, QK8, DOT, SEP)
    if DUAL:
        m1, l1, acc1 = _new(split, b, seq, j, q1, m1, l1, acc1, hp0 + 1, c, scale, k_scale, qkv_ptr, cs_ptr,
                            kc_ptr, vc_ptr, vs_ptr, loc_ptr, HQ, HKV, HPP, L, L_PAD, MG, D, ROPE, S, QK8, DOT, SEP)

    if WINDOW > 0:
        base = tl.maximum(seq - (WINDOW - 1), 0)
    else:
        base = seq * 0
    if BALANCE:
        # Each request's own context in S equal tile-aligned chunks: every program does the same work.
        chunk = tl.cdiv(tl.cdiv(seq - base, S), BLOCK_N) * BLOCK_N
    else:
        chunk = CHUNK
    lo = base + split * chunk
    hi = tl.minimum(lo + chunk, seq)
    pt_row = pt_ptr + b.to(tl.int64) * MAX_PAGES
    n = tl.arange(0, BLOCK_N)
    # Whole tiles need no bounds masks; full layers need no masks at all there. The tail is masked.
    # Only for single-token decode: with a verify block's 64 rows the second copy costs registers.
    if L == 1 or PEEL:
        full_end = lo + tl.maximum(hi - lo, 0) // BLOCK_N * BLOCK_N
    else:
        full_end = lo
    for start in tl.range(lo, full_end, BLOCK_N, num_stages=STAGES):
        m_i, l_i, acc = _tile(start + n, hi, pt_row, kc_ptr, vc_ptr, q, pos, c, m_i, l_i, acc, scale, hp0,
                              False, PAGE, ROW_W, D, WINDOW, QK8, DOT)
        if DUAL:
            m1, l1, acc1 = _tile(start + n, hi, pt_row, kc_ptr, vc_ptr, q1, pos, c, m1, l1, acc1, scale, hp0 + 1,
                                 False, PAGE, ROW_W, D, WINDOW, QK8, DOT)
    for start in tl.range(full_end, hi, BLOCK_N, num_stages=STAGES):
        m_i, l_i, acc = _tile(start + n, hi, pt_row, kc_ptr, vc_ptr, q, pos, c, m_i, l_i, acc, scale, hp0,
                              True, PAGE, ROW_W, D, WINDOW, QK8, DOT)
        if DUAL:
            m1, l1, acc1 = _tile(start + n, hi, pt_row, kc_ptr, vc_ptr, q1, pos, c, m1, l1, acc1, scale, hp0 + 1,
                                 True, PAGE, ROW_W, D, WINDOW, QK8, DOT)

    _finish(split, b, j, m_i, l_i, acc, hp0, hp, rr, r_ok, q_ok, c, pacc_ptr, pml_ptr, HKV, MG, D, S, SEP)
    if DUAL:
        _finish(split, b, j, m1, l1, acc1, hp0 + 1, hp, rr, r_ok, q_ok, c, pacc_ptr, pml_ptr, HKV, MG, D, S, SEP)

    if FUSE:
        # The last of this (request, head pair)'s S programs to finish merges the partials and
        # writes the output, then resets the counter for the next call.
        cnt = cnt_ptr + b * (HKV // HPP) + hp0 // HPP
        if tl.atomic_add(cnt, 1, sem="acq_rel") == S - 1:
            first = ((b * HKV + hp0 + hp) * S) * MG + rr
            # All splits' maxima and sums at once: [S_PAD, M_PAD].
            si = tl.arange(0, S_PAD)
            at = first[None, :] + si[:, None] * MG
            ok2 = (si < S)[:, None] & r_ok[None, :]
            m = tl.load(pml_ptr + at * 2, mask=ok2, other=-1.0e30, cache_modifier=".cg")
            m_max = tl.max(m, 0)
            denom = tl.sum(tl.exp2(m - m_max[None, :]) * tl.load(pml_ptr + at * 2 + 1, mask=ok2, other=0.0,
                                                                  cache_modifier=".cg"), 0)
            # Unrolled, so every split's accumulator load is in flight at once.
            o = tl.zeros([M_PAD, C], tl.float32)
            for i in tl.static_range(S):
                at1 = first + i * MG
                w = tl.exp2(tl.load(pml_ptr + at1 * 2, mask=r_ok, other=-1.0e30, cache_modifier=".cg") - m_max)
                o += w[:, None] * tl.load(pacc_ptr + at1[:, None] * D + (c % D)[None, :], mask=q_ok, other=0.0,
                                          cache_modifier=".cg").to(tl.float32)
            _norm_gate_store(o / tl.where(r_ok, denom, 1.0)[:, None] * tl.load(vs_ptr), qkv_ptr, out_ptr,
                             b * L + j, (hp0 + hp) * G + rr % G, c, q_ok, HQ, HKV, D)
            tl.atomic_xchg(cnt, 0)


@triton.jit
def _new_tokens(qkv_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, seq_ptr, loc_ptr, pacc_ptr, pml_ptr, qk_scale,
                HQ: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr, L: tl.constexpr, MG: tl.constexpr,
                L_PAD: tl.constexpr, D: tl.constexpr, S: tl.constexpr, DOT: tl.constexpr, NID: tl.constexpr = 0,
                NBASE: tl.constexpr = 0):
    """For _gw_verify: one program per (request, KV head) stores the block's new keys and values as FP8
    and attends them causally, into partial slot S (of S + 1)."""
    b = tl.program_id(0)
    hp0 = tl.program_id(1)
    gdc_launch_dependents()
    QKV_W: tl.constexpr = (2 * HQ + 2 * HKV) * D
    seq = tl.load(seq_ptr + b)
    c = tl.arange(0, D)
    rr = tl.arange(0, MG)
    j = rr // G
    q_off = (b * L + j)[:, None] * QKV_W + (hp0 * G + rr % G)[:, None] * D + c[None, :]
    q_ok = rr[:, None] < MG
    q = _norm_rope(qkv_ptr, q_off, q_ok, c, seq + j, qkv_ptr, False, D, 1).to(tl.float8e4nv)
    k_scale = tl.load(ks_ptr)
    m_i = tl.full([MG], -1.0e30, tl.float32)
    l_i = tl.zeros([MG], tl.float32)
    acc = tl.zeros([MG, D], tl.float32)
    m_i, l_i, acc = _new(0, b, seq, j, q, m_i, l_i, acc, hp0, c, qk_scale * k_scale, k_scale, qkv_ptr, qkv_ptr,
                         kc_ptr, vc_ptr, vs_ptr, loc_ptr, HQ, HKV, 1, L, L_PAD, MG, D, False, S, True, DOT)
    if NID:
        # Balanced splits (gbal): partials [HKV, NID, MG, D], this request's new tokens at id NBASE + b.
        _finish(NBASE + b, 0, j, m_i, l_i, acc, hp0, 0, rr, rr < MG, q_ok, c, pacc_ptr, pml_ptr, HKV, MG, D, NID)
    else:
        _finish(S, b, j, m_i, l_i, acc, hp0, 0, rr, rr < MG, q_ok, c, pacc_ptr, pml_ptr, HKV, MG, D, S + 1)


@triton.jit
def _norm_gate_store(o, qkv_ptr, out_ptr, tok, head, c, ok, HQ: tl.constexpr, HKV: tl.constexpr,
                     D: tl.constexpr):
    """out[tok, head] = rmsnorm(o) * sigmoid(gate) for [R, C] rows holding one head's D values each
    (the rest masked out and 0)."""
    QKV_W: tl.constexpr = (2 * HQ + 2 * HKV) * D
    o = o * tl.rsqrt(tl.sum(o * o, 1) / D + 1e-5)[:, None]
    col = head[:, None] * D + (c % D)[None, :]
    gate = tl.load(qkv_ptr + tok[:, None] * QKV_W + (HQ + 2 * HKV) * D + col, mask=ok, other=0.0).to(tl.float32)
    tl.store(out_ptr + tok[:, None] * HQ * D + col, (o * tl.sigmoid(gate)).to(tl.bfloat16), mask=ok)


@triton.jit
def _combine(qkv_ptr, pacc_ptr, pml_ptr, vs_ptr, out_ptr,
             HQ: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr, L: tl.constexpr, M: tl.constexpr,
             D: tl.constexpr, S: tl.constexpr, S_PAD: tl.constexpr, SB: tl.constexpr, PDL: tl.constexpr):
    tok = tl.program_id(0)
    head = tl.program_id(1)
    b = tok // L
    h = head // G
    QKV_W: tl.constexpr = (2 * HQ + 2 * HKV) * D
    d = tl.arange(0, D)
    gate = tl.load(qkv_ptr + tok * QKV_W + (HQ + 2 * HKV) * D + head * D + d).to(tl.float32)
    if PDL:
        # Launched early (programmatic dependent launch): wait here for _attn_split's partials.
        gdc_wait()
    # This head's row in each split's partial: [S] maxima and sums, [S, D] accumulators.
    row = (b * HKV + h) * S * M + (tok % L) * G + head % G
    s = tl.arange(0, S_PAD)
    m = tl.load(pml_ptr + (row + s * M) * 2, mask=s < S, other=-1.0e30)
    l = tl.load(pml_ptr + (row + s * M) * 2 + 1, mask=s < S, other=0.0)
    m_max = tl.max(m, 0)
    denom = tl.sum(tl.exp2(m - m_max) * l, 0)
    acc = tl.zeros([D], tl.float32)
    sb = tl.arange(0, SB)
    for i in range(0, S, SB):
        si = i + sb
        w = tl.exp2(tl.load(pml_ptr + (row + si * M) * 2, mask=si < S, other=-1.0e30) - m_max)
        a = tl.load(pacc_ptr + (row + si * M)[:, None] * D + d[None, :], mask=(si < S)[:, None],
                    other=0.0).to(tl.float32)
        acc += tl.sum(w[:, None] * a, 0)
    o = acc / denom * tl.load(vs_ptr)
    o = o * tl.rsqrt(tl.sum(o * o, 0) / D + 1e-5)
    tl.store(out_ptr + tok * HQ * D + head * D + d, (o * tl.sigmoid(gate)).to(tl.bfloat16))


@triton.jit
def _combine_hb(qkv_ptr, pacc_ptr, pml_ptr, vs_ptr, out_ptr,
                HQ: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr, L: tl.constexpr, M: tl.constexpr,
                D: tl.constexpr, S: tl.constexpr, S_PAD: tl.constexpr, SB: tl.constexpr, HB: tl.constexpr,
                PDL: tl.constexpr):
    """_combine for HB adjacent query heads of one KV head per program: their partial rows are
    contiguous in every split, so each split's [HB, D] block is one contiguous read. Faster on
    sliding layers (fewer, fuller programs); full layers keep _combine."""
    tok = tl.program_id(0)
    head0 = tl.program_id(1) * HB
    b = tok // L
    h = head0 // G
    QKV_W: tl.constexpr = (2 * HQ + 2 * HKV) * D
    d = tl.arange(0, D)
    hb = tl.arange(0, HB)
    col = (head0 + hb)[:, None] * D + d[None, :]
    gate = tl.load(qkv_ptr + tok * QKV_W + (HQ + 2 * HKV) * D + col).to(tl.float32)
    if PDL:
        gdc_wait()
    row = (b * HKV + h) * S * M + (tok % L) * G + head0 % G + hb  # [HB] rows in split 0
    s = tl.arange(0, S_PAD)
    at = row[None, :] + (s * M)[:, None]
    m = tl.load(pml_ptr + at * 2, mask=(s < S)[:, None], other=-1.0e30)
    l = tl.load(pml_ptr + at * 2 + 1, mask=(s < S)[:, None], other=0.0)
    m_max = tl.max(m, 0)
    denom = tl.sum(tl.exp2(m - m_max[None, :]) * l, 0)
    acc = tl.zeros([HB, D], tl.float32)
    sb = tl.arange(0, SB)
    for i in range(0, S, SB):
        si = i + sb
        at1 = row[None, :] + (si * M)[:, None]
        ok = (si < S)[:, None]
        w = tl.exp2(tl.load(pml_ptr + at1 * 2, mask=ok, other=-1.0e30) - m_max[None, :])
        a = tl.load(pacc_ptr + at1[:, :, None] * D + d[None, None, :], mask=ok[:, :, None], other=0.0).to(tl.float32)
        acc += tl.sum(w[:, :, None] * a, 0)
    o = acc / denom[:, None] * tl.load(vs_ptr)
    o = o * tl.rsqrt(tl.sum(o * o, 1) / D + 1e-5)[:, None]
    tl.store(out_ptr + tok * HQ * D + col, (o * tl.sigmoid(gate)).to(tl.bfloat16))


@triton.jit
def _combine_bal(qkv_ptr, seq_ptr, pacc_ptr, pml_ptr, vs_ptr, out_ptr,
                 HQ: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr, L: tl.constexpr, M: tl.constexpr,
                 D: tl.constexpr, B: tl.constexpr, B_PAD: tl.constexpr, P: tl.constexpr, NID: tl.constexpr,
                 BN: tl.constexpr, SB: tl.constexpr, HB: tl.constexpr):
    """_combine_hb for balanced splits (_gw_bal): request b's partials are at ids p + b for the programs p whose
    tile range meets b's tiles, and its new tokens at P + B + b."""
    tok = tl.program_id(0)
    head0 = tl.program_id(1) * HB
    b = tok // L
    h = head0 // G
    QKV_W: tl.constexpr = (2 * HQ + 2 * HKV) * D
    d = tl.arange(0, D)
    hb = tl.arange(0, HB)
    col = (head0 + hb)[:, None] * D + d[None, :]
    gate = tl.load(qkv_ptr + tok * QKV_W + (HQ + 2 * HKV) * D + col).to(tl.float32)
    idx = tl.arange(0, B_PAD)
    tiles = (tl.load(seq_ptr + idx, mask=idx < B, other=0).to(tl.int32) + (BN - 1)) // BN
    tt = tl.maximum(tl.sum(tiles, 0), 1)
    excl = tl.sum(tl.where(idx < b, tiles, 0), 0)
    incl = excl + tl.sum(tl.where(idx == b, tiles, 0), 0)
    # Programs p with start(p) < incl, end(p) > excl, end(p) > start(p) (a superset range, then the test).
    p_lo = tl.maximum((excl + 1) * P // tt - 1, 0)
    p_hi = tl.minimum((incl * P) // tt + 1, P - 1)
    gdc_wait()
    row = (h * NID + b) * M + (tok % L) * G + head0 % G + hb  # [HB] rows at id b (p = 0)
    nrow = row + (P + B) * M
    m_max = tl.load(pml_ptr + nrow * 2)
    sb = tl.arange(0, SB)
    for i in range(p_lo, p_hi + 1, SB):
        pp = i + sb
        st = pp * tt // P
        en = (pp + 1) * tt // P
        ok = ((pp <= p_hi) & (st < incl) & (en > excl) & (en > st))[:, None]
        at = row[None, :] + (pp * M)[:, None]
        m_max = tl.maximum(m_max, tl.max(tl.load(pml_ptr + at * 2, mask=ok, other=-1.0e30), 0))
    wn = tl.exp2(tl.load(pml_ptr + nrow * 2) - m_max)
    denom = wn * tl.load(pml_ptr + nrow * 2 + 1)
    acc = wn[:, None] * tl.load(pacc_ptr + nrow[:, None] * D + d[None, :]).to(tl.float32)
    for i in range(p_lo, p_hi + 1, SB):
        pp = i + sb
        st = pp * tt // P
        en = (pp + 1) * tt // P
        ok = ((pp <= p_hi) & (st < incl) & (en > excl) & (en > st))[:, None]
        at = row[None, :] + (pp * M)[:, None]
        w = tl.exp2(tl.load(pml_ptr + at * 2, mask=ok, other=-1.0e30) - m_max[None, :])
        denom += tl.sum(w * tl.load(pml_ptr + at * 2 + 1, mask=ok, other=0.0), 0)
        a = tl.load(pacc_ptr + at[:, :, None] * D + d[None, None, :], mask=ok[:, :, None], other=0.0).to(tl.float32)
        acc += tl.sum(w[:, :, None] * a, 0)
    o = acc / denom[:, None] * tl.load(vs_ptr)
    o = o * tl.rsqrt(tl.sum(o * o, 1) / D + 1e-5)[:, None]
    tl.store(out_ptr + tok * HQ * D + col, (o * tl.sigmoid(gate)).to(tl.bfloat16))


def _plan(cfg, max_pages, device):
    """Launch parameters from shapes only, tuned on an H100 over the scored shapes: splits per
    (request, KV head group), tile size, warps, pipeline stages, KV heads per program (hpp),
    whether the last program merges the splits (fuse), and whether QK^T runs in FP8 (qk8:
    queries rounded to E4M3, as FA3 does with an FP8 cache)."""
    batch, length, hkv = cfg["batch"], cfg["q_len"], cfg["num_kv_heads"]
    heads = batch * hkv
    pairs = heads // 2
    if cfg["window"] and length == 1 and pairs >= 12 and HAVE_GLUON and USE_GSLIDE and hkv % 2 == 0 \
            and cfg["num_q_heads"] // hkv == 8 and cfg["head_dim"] == 64:
        # Batched sliding decode on the Gluon one-warp kernel (FP8 QK^T mma; a dedicated program per KV head
        # pair for the new token, so the S key splits stay tile-aligned), up to ~800 programs (H100 decode
        # 16/32 sliding 10.6/15.6 -> 9.3/13.9 us, B64 27.7 -> 23.2; at batch <= 8 it loses by 0.2-1.8 us).
        # Off: on the H200 (check f3e099c1) decode 16/32 sliding went 10.4/14.6 -> 13.6/15.4 us.
        splits = 32 if pairs * 32 <= 800 else (16 if pairs * 16 <= 800 else 8)
        return dict(gdec=True, sep=True, splits=splits, block_n=32, stages=2, pdt="bf16", hb=2)
    if cfg["window"]:
        # Sliding layers: at most 2,047 cached keys; latency-bound.
        # The merge takes 2 query heads per one-warp program (hb, cw) and every split in one pass (sb):
        # each further pass is another serial L2 round trip on the critical path.
        splits = 32 if heads <= 4 else (16 if heads <= 32 and length == 1 else 8)
        # Batched verify blocks write BF16 partial accumulators (H100 verify 16/32 sliding -0.5 us); decode
        # and batch 1 keep FP32 (BF16 there is +0.1-0.3 us).
        pdt = "bf16" if length > 1 and heads > 4 else "f32"
        if heads <= 32:
            # tsep (batch 1-2): the block's new tokens in their own program instead of split 0, the critical path,
            # with 2 stages (one tile per split). H100 paired: verify 1 sliding tsep -0.14 us (8/9 rounds twice),
            # + stages 2 -0.11 (9/15); decode 1 sliding tsep alone and stages 2 alone ties, both -0.35 (15/15).
            # Batched sizes lose with tsep (+0.6-2.7 us).
            tiny = heads <= 4
            return dict(splits=splits, block_n=64, warps=4, stages=2 if tiny else (5 if length > 1 else 3),
                        hpp=1, fuse=False, qk8=True, hb=2, sb=splits, cw=1, pdt=pdt, tsep=tiny)
        # Large batches: smaller, deeply pipelined programs. (hpp 2 for decode won on the H100, decode 32 sliding
        # 16.0 -> 15.4 us, but is unchecked on the H200; reverted with session 18's decode 1 full plan.)
        return dict(splits=splits, block_n=32, warps=2 if length == 1 else 4, stages=5, hpp=1, fuse=False,
                    qk8=True, peel=length > 1, hb=2, sb=splits, cw=1, pdt=pdt)
    # Full layers' Triton paths write BF16 partial accumulators (unnormalized; FP16 could overflow): H100
    # decode 16/32 full -1.0/-1.5 us, verify 1 full -0.4. The Gluon verify path keeps FP32 (BF16 +1.5 us).
    if heads == 2 and length == 1 and HAVE_GLUON and USE_GDEC and USE_GD1 and cfg["num_q_heads"] // hkv == 8 \
            and cfg["head_dim"] == 64:
        # Batch-1 decode full on the Gluon one-warp kernel: 511 key splits + the new token's slot (512 BF16
        # partials, about 4 per SM), merged in one pass by 8-warp programs. H100 17.5 -> 15.7 us (attention
        # alone 15.3 -> 12.4, near the 11.5 us gather floor; the 512-way merge costs 3.2 us vs 2.2).
        return dict(gdec=True, splits=511, block_n=32, stages=2, pdt="bf16", csb=512, cw=8)
    if heads <= (2 if length == 1 else 4):
        # (Session 18's hpp 2 / block_n 64 for decode 1 full won on the H100, 18.0 -> 17.4 us, but lost on the
        # H200: check f3e099c1 decode 1 full 15.2 -> 17.6 us.)
        # tsep as on sliding layers (verify 1 full paired -0.11 us, 9/11 rounds), with 127 splits so the merge sees
        # at most 128 slots in one pass (-0.08, 8/11).
        return dict(splits=127 if length > 1 else 128, block_n=128, warps=4, stages=6, hpp=1, fuse=False, qk8=True, pdt="bf16",
                    tsep=length > 1)
    if length == 1:
        if HAVE_GLUON and USE_GDEC and hkv % 2 == 0 and cfg["num_q_heads"] // hkv == 8 and cfg["head_dim"] == 64:
            # Gluon one-warp programs (FP8 QK^T mma, new token in the last split): 1024 programs (one wave),
            # S + 1 partial slots a power of two for the merge, BF16 partials.
            # Merge in one-warp programs up to 128 slots (H100 batch 8-48 -1.0..-2.1 us, decode 16/32 full
            # -1.7/-1.7; 256-512 slots lose 0.7-1.0 us). Likely residency: 8 attention programs at 242 registers
            # leave an SM 2048 registers for PDL-launched merge programs (capping the merge's registers loses).
            splits = max(4, 2048 // heads - 1)
            return dict(gdec=True, splits=splits, block_n=32, stages=2, pdt="bf16", **({"cw": 1} if splits < 128 else {}))
        # Many small one-warp programs with deep pipelines keep the most scattered row reads in flight.
        return dict(splits=max(8, 2048 // heads), block_n=16, warps=1, stages=5 if heads <= 32 else 6, hpp=2,
                    fuse=False, qk8=True, pdt="bf16")
    # Verify blocks: each program runs both KV heads as two independent 64-row chains (dual), which
    # hides latency better than more programs; 3 fit per SM (160 registers), or 4 when capped at 128
    # (no spills). The splits fill exactly one wave, and each request's context is cut into equal
    # chunks (balance) so none idles. 4 per SM wins from 6 pairs up (batch 32: 303 -> 292 us, 64:
    # 593 -> 568 on an H100); below that only where it fills the wave better (batch 3-4 lose 2-4%).
    pairs = batch * max(1, hkv // 2)
    sms = _num_sms(device)
    if HAVE_GLUON and pairs >= 6 and hkv % 2 == 0 and length * cfg["num_q_heads"] // hkv == 64 \
            and cfg["head_dim"] == 64:
        # The warp-specialized Gluon kernel: 2 programs per SM (384 threads). One wave of programs unless
        # two fill the slots clearly better (H100, ctx 80K: batch 16 155 -> 145 us with 16 splits instead
        # of 32, batch 12 124 -> 116; batch 40 keeps two waves, 344 vs 332).
        slots = 2 * sms
        fill = {w: (w * slots // pairs) * pairs / (w * slots) for w in (1, 2)}
        waves = 2 if fill[2] > fill[1] + 0.05 else 1
        splits = max(2, waves * slots // pairs)
        # The merge: with up to 32 partials, one-warp programs over 2 heads (H100 verify 16 full 139 -> 137.5 us).
        merge = dict(hb=2, cw=1) if splits + 1 <= 32 and (cfg["num_q_heads"] // hkv) % 2 == 0 else {}
        # gfuse: the merge runs in the same grid (each program merges a slice of rows once its request's
        # programs have all stored; H100 verify 16/32 full 136.8 -> 136.0 / 257.2 -> 256.2 us). The separate
        # merge (hb/cw) remains for two-wave plans.
        # gbal (from 16 pairs): the programs instead share the batch's key tiles equally (ranges may span
        # requests; the agentic lengths differ by up to 15%), as many as fill the slots; merged by _combine_bal.
        # H100 all-L2 (compute-bound, as the H200 mostly is) verify 16/32 full 110.0/212.4 -> 104.1/195.9 us,
        # real 136.4/255.4 -> 138.8/250.7, batch 40/48/64 real -8.8/-7.1/-8.4. Below 16 pairs the lost in-grid
        # merge (gfuse) costs more than the balance gains (batch 8: +4 us real).
        return dict(gluon=True, splits=splits, block_n=64, stages=3, lag=1, gpeel=True, pw=4,
                    pregs=40, cregs=96, maxnreg=80, lsum=True, gfuse=True, gbal=USE_GBAL and pairs >= 16, **merge)
    fill = {per: max(4, per * sms // pairs) * pairs / (per * sms) for per in (3, 4)}
    per = 4 if pairs >= 6 or fill[4] > fill[3] else 3
    return dict(splits=max(4, per * sms // pairs), block_n=32, warps=4, stages=5, hpp=1, fuse=False,
                qk8=True, peel=True, dual=True, balance=True, maxnreg=128 if per == 4 else None, pdt="bf16")


def kernel_fn(data: Inputs) -> torch.Tensor:
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    group = hq // hkv
    qkv = data["qkv"]
    device = qkv.device
    page_table = data["page_table"]
    max_pages = page_table.shape[1]
    plan = _plan(cfg, max_pages, device)
    if plan.get("gluon"):
        return _verify_gluon(data, cfg, plan)
    if plan.get("gdec"):
        return _decode_gluon(data, cfg, plan)
    span = max_pages * cfg["page_size"]
    if cfg["window"]:
        span = min(span, cfg["window"])
    block_n = plan["block_n"]
    chunk = triton.cdiv(triton.cdiv(span, plan["splits"]), block_n) * block_n
    splits = plan["splits"] if plan.get("balance") else triton.cdiv(span, chunk)
    rows = length * group
    hpp = plan["hpp"] if hkv % plan["hpp"] == 0 else 1
    dual = plan.get("dual", False) and hpp == 1 and hkv % 2 == 0
    rope = bool(cfg["rope_theta"])
    cs = _cos_sin(cfg["rope_theta"], d, device) if rope else qkv

    fuse = plan["fuse"] and splits <= 64 and not dual  # the fused merge unrolls over splits
    # tsep: the block's new tokens in an extra program (partial slot S) instead of split 0.
    sep = int(bool(plan.get("tsep")) and not fuse)
    pacc = torch.empty((batch, hkv, splits + sep, rows, d), dtype=_PDT[plan.get("pdt", "f32")], device=device)
    pml = torch.empty((batch, hkv, splits + sep, rows, 2), dtype=torch.float32, device=device)
    out = torch.empty((batch, length, hq * d), dtype=torch.bfloat16, device=device)
    _attn_split[(splits + sep, batch, hkv // (2 if dual else hpp))](
        qkv, cs, data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"], page_table,
        data["seq_lens"], data["cache_loc"], pacc, pml, _counters(batch * hkv, device), out,
        cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E,
        HQ=hq, HKV=hkv, HPP=hpp, G=group, L=length, MG=rows, M_PAD=max(16, triton.next_power_of_2(hpp * rows)),
        L_PAD=max(16, triton.next_power_of_2(length)), D=d, PAGE=cfg["page_size"], MAX_PAGES=max_pages,
        WINDOW=cfg["window"], ROPE=rope, S=splits, CHUNK=chunk, BLOCK_N=block_n, STAGES=plan["stages"],
        FUSE=fuse, S_PAD=triton.next_power_of_2(splits), QK8=plan["qk8"], PDL=USE_PDL and not fuse, DOT=DOT_DTYPE, PEEL=plan.get("peel", False), DUAL=dual, BALANCE=plan.get("balance", False), SEP=bool(sep), num_warps=plan["warps"],
        **({"maxnreg": plan["maxnreg"]} if plan.get("maxnreg") else {}))
    if fuse:
        return out
    splits += sep
    hb = plan.get("hb", 1) if group % plan.get("hb", 1) == 0 else 1
    if hb > 1:
        sb = plan.get("sb", max(1, min(triton.next_power_of_2(splits), 32 // hb)))
        if sep:
            sb = triton.next_power_of_2(max(sb, splits))
        _combine_hb[(batch * length, hq // hb)](
            qkv, pacc, pml, data["v_scale"], out,
            HQ=hq, HKV=hkv, G=group, L=length, M=rows, D=d, S=splits, S_PAD=triton.next_power_of_2(splits),
            SB=sb, HB=hb, PDL=USE_PDL, num_warps=plan.get("cw", 4), launch_pdl=USE_PDL)
        return out
    _combine[(batch * length, hq)](
        qkv, pacc, pml, data["v_scale"], out,
        HQ=hq, HKV=hkv, G=group, L=length, M=rows, D=d, S=splits, S_PAD=triton.next_power_of_2(splits),
        # Up to 128 splits in one pass, so every partial's load is in flight at once.
        SB=min(128, triton.next_power_of_2(splits)), PDL=USE_PDL, num_warps=4 if splits > 32 else 2,
        launch_pdl=USE_PDL)
    return out


def _verify_gluon(data, cfg, plan):
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    group = hq // hkv
    rows = length * group
    qkv = data["qkv"]
    device = qkv.device
    page_table = data["page_table"]
    splits = plan["splits"]
    out = torch.empty((batch, length, hq * d), dtype=torch.bfloat16, device=device)
    qk_scale = cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E
    if plan.get("gbal"):
        return _verify_bal(data, cfg, plan, out, qk_scale)
    pacc = torch.empty((batch, hkv, splits + 1, rows, d), dtype=_PDT[plan.get("pdt", "f32")], device=device)
    pml = torch.empty((batch, hkv, splits + 1, rows, 2), dtype=torch.float32, device=device)
    # The in-grid merge (gfuse) needs every program resident at once (one wave).
    pairs = batch * (hkv // 2)
    fuse = plan.get("gfuse", False) and splits * pairs <= 2 * _num_sms(device) \
        and splits < 0xFFFF
    rs = triton.cdiv(2 * rows, splits)
    _new_tokens[(batch, hkv)](
        qkv, data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"], data["seq_lens"], data["cache_loc"],
        pacc, pml, qk_scale, HQ=hq, HKV=hkv, G=group, L=length, MG=rows, L_PAD=max(16, triton.next_power_of_2(length)),
        D=d, S=splits, DOT=DOT_DTYPE, num_warps=4)
    _gw_verify[(splits, batch, hkv // 2)](
        qkv, data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"], page_table, data["seq_lens"],
        data["cache_loc"], pacc, pml, qk_scale,
        HQ=hq, HKV=hkv, G=group, L=length, D=d, PAGE=cfg["page_size"], MAX_PAGES=page_table.shape[1], S=splits,
        BN=plan["block_n"], NS=plan["stages"], PW=plan["pw"], PREGS=plan["pregs"], CREGS=plan["cregs"],
        LAG=plan.get("lag", 0), PEEL=plan.get("gpeel", False), LSUM=plan.get("lsum", False), FUSE=fuse, cnt_ptr=_counters(pairs, device, "g"), out_ptr=out, RS=rs,
        R_PAD=triton.next_power_of_2(rs), num_warps=4, maxnreg=plan["maxnreg"], launch_pdl=True)
    if fuse:
        return out
    s = splits + 1
    hb = plan.get("hb", 1)
    if hb > 1:
        _combine_hb[(batch * length, hq // hb)](
            qkv, pacc, pml, data["v_scale"], out,
            HQ=hq, HKV=hkv, G=group, L=length, M=rows, D=d, S=s, S_PAD=triton.next_power_of_2(s),
            SB=triton.next_power_of_2(s), HB=hb, PDL=USE_PDL, num_warps=plan.get("cw", 1), launch_pdl=USE_PDL)
        return out
    _combine[(batch * length, hq)](
        qkv, pacc, pml, data["v_scale"], out,
        HQ=hq, HKV=hkv, G=group, L=length, M=rows, D=d, S=s, S_PAD=triton.next_power_of_2(s),
        SB=min(128, triton.next_power_of_2(s)), PDL=USE_PDL, num_warps=plan.get("cw", 4 if s > 32 else 2),
        launch_pdl=USE_PDL)
    return out


def _verify_bal(data, cfg, plan, out, qk_scale):
    """_verify_gluon with balanced splits (plan gbal): the programs of each KV head pair index share the batch's
    key tiles equally; partials [HKV, NID, rows, D] by id (see _gw_bal); merged by _combine_bal."""
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    group = hq // hkv
    rows = length * group
    qkv = data["qkv"]
    device = qkv.device
    page_table = data["page_table"]
    # Programs per KV head pair: as many as fill the slots (2 per SM), or the plan's splits per request.
    progs = 2 * _num_sms(device) // (hkv // 2) if plan.get("fill", True) else plan["splits"] * batch
    nid = progs + 2 * batch
    b_pad = plan.get("bpad", max(32, triton.next_power_of_2(batch)))
    pacc = torch.empty((hkv, nid, rows, d), dtype=_PDT[plan.get("pdt", "f32")], device=device)
    pml = torch.empty((hkv, nid, rows, 2), dtype=torch.float32, device=device)
    _new_tokens[(batch, hkv)](
        qkv, data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"], data["seq_lens"], data["cache_loc"],
        pacc, pml, qk_scale, HQ=hq, HKV=hkv, G=group, L=length, MG=rows, L_PAD=max(16, triton.next_power_of_2(length)),
        D=d, S=1, DOT=DOT_DTYPE, NID=nid, NBASE=progs + batch, num_warps=4)
    _gw_verify[(progs, 1, hkv // 2)](
        qkv, data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"], page_table, data["seq_lens"],
        data["cache_loc"], pacc, pml, qk_scale,
        HQ=hq, HKV=hkv, G=group, L=length, D=d, PAGE=cfg["page_size"], MAX_PAGES=page_table.shape[1], S=progs,
        BN=plan["block_n"], NS=plan["stages"], PW=plan["pw"], PREGS=plan["pregs"], CREGS=plan["cregs"],
        LAG=plan.get("lag", 1), BAL=True, B=batch, B_PAD=b_pad, NID=nid, NP=progs, num_warps=4, maxnreg=plan["maxnreg"],
        launch_pdl=True)
    hb = plan.get("hb", 2)
    _combine_bal[(batch * length, hq // hb)](
        qkv, data["seq_lens"], pacc, pml, data["v_scale"], out,
        HQ=hq, HKV=hkv, G=group, L=length, M=rows, D=d, B=batch, B_PAD=b_pad, P=progs, NID=nid,
        BN=plan["block_n"], SB=plan.get("sb", triton.next_power_of_2(progs // batch + 3)), HB=hb,
        num_warps=plan.get("cw", 1), launch_pdl=USE_PDL)
    return out


def _decode_gluon(data, cfg, plan):
    batch, hq, hkv, d = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    group = hq // hkv
    qkv = data["qkv"]
    device = qkv.device
    page_table = data["page_table"]
    splits = plan["splits"]
    s = splits + 1
    pacc = torch.empty((batch, hkv, s, group, d), dtype=_PDT[plan.get("pdt", "f32")], device=device)
    pml = torch.empty((batch, hkv, s, group, 2), dtype=torch.float32, device=device)
    out = torch.empty((batch, 1, hq * d), dtype=torch.bfloat16, device=device)
    sep = plan.get("sep", False)
    _gd_full[(splits + sep, batch, hkv // 2)](
        qkv, data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"], page_table, data["seq_lens"],
        data["cache_loc"], pacc, pml, cfg["sm_scale"] * cfg["qk_scale_factor"] * LOG2E,
        _cos_sin(cfg["rope_theta"], d, device) if cfg["rope_theta"] else qkv,
        HQ=hq, HKV=hkv, G=group, D=d, PAGE=cfg["page_size"], MAX_PAGES=page_table.shape[1], S=splits,
        BN=plan["block_n"], NS=plan["stages"], TPV=plan.get("tpv", False), WINDOW=cfg["window"],
        ROPE=bool(cfg["rope_theta"]), SEP=sep, num_warps=1)
    hb = plan.get("hb", 1)
    if hb > 1:
        _combine_hb[(batch, hq // hb)](
            qkv, pacc, pml, data["v_scale"], out,
            HQ=hq, HKV=hkv, G=group, L=1, M=group, D=d, S=s, S_PAD=triton.next_power_of_2(s),
            SB=triton.next_power_of_2(s), HB=hb,
            PDL=USE_PDL, num_warps=plan.get("cw", 1), launch_pdl=USE_PDL)
        return out
    _combine[(batch, hq)](
        qkv, pacc, pml, data["v_scale"], out,
        HQ=hq, HKV=hkv, G=group, L=1, M=group, D=d, S=s, S_PAD=triton.next_power_of_2(s),
        SB=plan.get("csb", min(128, triton.next_power_of_2(s))), PDL=USE_PDL,
        num_warps=plan.get("cw", 4 if s > 32 else 2), launch_pdl=USE_PDL)
    return out

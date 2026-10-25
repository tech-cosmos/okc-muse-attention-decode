"""agent2: _combine_nt = _combine_hb + the verify block's new tokens (norm/RoPE/E4M3, cache store, causal attention in
FP32) computed before gdc_wait, so they overlap the attention kernel. Imported by nvs.py (NV_NEW=0 path)."""
import sys
sys.path.insert(0, "/home/daytona/repo2")
import triton, triton.language as tl
from submission import _norm_rope, gdc_wait


@triton.jit
def _combine_nt(qkv_ptr, cs_ptr, kc_ptr, vc_ptr, ks_ptr, vs_ptr, seq_ptr, loc_ptr, pacc_ptr, pml_ptr, out_ptr,
                qk_scale, HQ: tl.constexpr, HKV: tl.constexpr, G: tl.constexpr, L: tl.constexpr, M: tl.constexpr,
                D: tl.constexpr, S: tl.constexpr, S_PAD: tl.constexpr, HB: tl.constexpr, PDL: tl.constexpr):
    tok = tl.program_id(0)
    head0 = tl.program_id(1) * HB
    b = tok // L
    jq = tok % L
    h = head0 // G
    QKV_W: tl.constexpr = (2 * HQ + 2 * HKV) * D
    ROW_W: tl.constexpr = HKV * D
    d = tl.arange(0, D)
    hb = tl.arange(0, HB)
    col = (head0 + hb)[:, None] * D + d[None, :]
    gate = tl.load(qkv_ptr + tok * QKV_W + (HQ + 2 * HKV) * D + col).to(tl.float32)
    seq = tl.load(seq_ptr + b)
    # Queries (as _attn_body: RMS norm, RoPE, E4M3) and the block's new keys / values of KV head h.
    q = _norm_rope(qkv_ptr, tok * QKV_W + col, hb[:, None] < HB, d, seq + jq + hb * 0, cs_ptr, True, D, 1)
    q = q.to(tl.float8e4nv).to(tl.float32)
    jn = tl.arange(0, L)
    kn_off = ((b * L + jn) * QKV_W + (HQ + h) * D)[:, None] + d[None, :]
    k_scale = tl.load(ks_ptr)
    v_scale = tl.load(vs_ptr)
    kn = _norm_rope(qkv_ptr, kn_off, jn[:, None] < L, d, seq + jn, cs_ptr, True, D, 1)
    kq = (kn / k_scale).to(tl.float8e4nv)
    vq = (tl.load(qkv_ptr + kn_off + HKV * D).to(tl.float32) / v_scale).to(tl.float8e4nv)
    if (jq == 0) & (head0 % G == 0):
        slot = tl.load(loc_ptr + b * L + jn) * ROW_W + h * D
        tl.store(kc_ptr + slot[:, None] + d[None, :], kq)
        tl.store(vc_ptr + slot[:, None] + d[None, :], vq)
    s = tl.sum(q[:, None, :] * kq.to(tl.float32)[None, :, :], 2) * (qk_scale * k_scale)  # [HB, L]
    s = tl.where((jn <= jq)[None, :], s, float("-inf"))
    m_n = tl.max(s, 1)
    p = tl.exp2(s - m_n[:, None])
    l_n = tl.sum(p, 1)
    acc_n = tl.sum(p[:, :, None] * vq.to(tl.float32)[None, :, :], 1)  # [HB, D]
    if PDL:
        gdc_wait()
    row = (b * HKV + h) * S * M + jq * G + head0 % G + hb
    si = tl.arange(0, S_PAD)
    at = row[None, :] + (si * M)[:, None]
    ok = (si < S)[:, None]
    m = tl.load(pml_ptr + at * 2, mask=ok, other=-1.0e30)
    l = tl.load(pml_ptr + at * 2 + 1, mask=ok, other=0.0)
    a = tl.load(pacc_ptr + at[:, :, None] * D + d[None, None, :], mask=ok[:, :, None], other=0.0)
    m_max = tl.maximum(tl.max(m, 0), m_n)
    w = tl.exp2(m - m_max[None, :])
    wn = tl.exp2(m_n - m_max)
    denom = tl.sum(w * l, 0) + wn * l_n
    acc = tl.sum(w[:, :, None] * a, 0) + wn[:, None] * acc_n
    o = acc / denom[:, None] * v_scale
    o = o * tl.rsqrt(tl.sum(o * o, 1) / D + 1e-5)[:, None]
    tl.store(out_ptr + tok * HQ * D + col, (o * tl.sigmoid(gate)).to(tl.bfloat16))

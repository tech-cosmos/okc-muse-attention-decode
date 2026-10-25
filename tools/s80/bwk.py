"""Session 80 H200 bandwidth diag kernels: dense / gathered / sorted 128-B row reads of the decode-32-full bytes."""
import torch, triton, triton.language as tl


@triton.jit
def _bw_read(kc, vc, idx, seq, out, B: tl.constexpr, B_PAD: tl.constexpr, IDX: tl.constexpr, LOCK: tl.constexpr,
             BN: tl.constexpr, ST: tl.constexpr):
    # Rows 0..T-1 (T = sum(seq)) of the list idx (IDX) or of the cache itself (dense), K and V, summed.
    pid = tl.program_id(0)
    P = tl.num_programs(0)
    bi = tl.arange(0, B_PAD)
    T = tl.sum(tl.load(seq + bi, mask=bi < B, other=0).to(tl.int64), 0)
    c = tl.arange(0, 128)
    n = tl.arange(0, BN)
    acc = tl.zeros([BN, 128], tl.float32)
    ntile = tl.cdiv(T, BN)
    if LOCK:
        # tile k*P + pid: every program sweeps memory together
        for k in tl.range(0, tl.cdiv(ntile - pid, P), num_stages=ST):
            s = (k * P + pid) * BN + n
            ok = s < T
            if IDX:
                row = tl.load(idx + s, mask=ok, other=0).to(tl.int64) * 128
            else:
                row = s * 128
            acc += tl.load(kc + row[:, None] + c[None, :], mask=ok[:, None]).to(tl.float32) \
                + tl.load(vc + row[:, None] + c[None, :], mask=ok[:, None]).to(tl.float32)
    else:
        t0 = pid * ntile // P
        t1 = (pid + 1) * ntile // P
        for k in tl.range(t0, t1, num_stages=ST):
            s = k * BN + n
            ok = s < T
            if IDX:
                row = tl.load(idx + s, mask=ok, other=0).to(tl.int64) * 128
            else:
                row = s * 128
            acc += tl.load(kc + row[:, None] + c[None, :], mask=ok[:, None]).to(tl.float32) \
                + tl.load(vc + row[:, None] + c[None, :], mask=ok[:, None]).to(tl.float32)
    tl.store(out + pid * 128 + c, tl.sum(acc, 0))


def bw_lists(page_table, seq_lens):
    """(request-order list, globally sorted list, per-request sorted list) of the valid rows; graph capturable."""
    B, MP = page_table.shape
    t = torch.arange(MP, device=page_table.device)
    valid = t[None, :] < seq_lens[:, None]
    big = torch.iinfo(torch.int32).max
    pos = torch.where(valid, torch.arange(B * MP, device=page_table.device).view(B, MP), B * MP)
    order = torch.sort(pos.flatten()).values.clamp(max=B * MP - 1)
    flat = page_table.flatten()
    rand = flat[order].contiguous()
    gsort = torch.sort(torch.where(valid, page_table, big).flatten()).values.contiguous()
    rsort = torch.sort(torch.where(valid, page_table, big), dim=1).values.flatten()[order].contiguous()
    return rand, gsort, rsort

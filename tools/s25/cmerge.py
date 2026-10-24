"""Microbench: cost of an in-cluster online-softmax merge of NC partials [M, C] (DSMEM reductions)."""
import sys, fcntl, torch
import triton.language.core as core
_orig = core._type_for_tuple_values
def _patched(values, fields=None):
    return _orig([core.constexpr(x) if isinstance(x, list) else x for x in values], fields)
core._type_for_tuple_values = _patched
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import cluster


@gluon.jit
def k(x_ptr, ml_ptr, out_ptr, NC: gl.constexpr, M: gl.constexpr, C: gl.constexpr, MERGE: gl.constexpr, L3: gl.constexpr, LB: gl.constexpr, P: gl.constexpr):
    L2: gl.constexpr = gl.SliceLayout(2, L3)
    s = gl.arange(0, NC, layout=gl.SliceLayout(1, L2))
    r = gl.arange(0, M, layout=gl.SliceLayout(0, L2))
    c = gl.arange(0, C, layout=gl.SliceLayout(0, gl.SliceLayout(1, L3)))
    pid = gl.program_id(0)
    base = pid * NC * M
    acc = gl.load(x_ptr + ((base + s[:, None] * M + r[None, :]) * C)[:, :, None] + c[None, None, :])
    m = gl.load(ml_ptr + (base + s[:, None] * M + r[None, :]) * 2)
    l = gl.load(ml_ptr + (base + s[:, None] * M + r[None, :]) * 2 + 1)
    if MERGE == 2:
        # Exchange through L2: own slice out, cluster barrier, everything back in a CTA-broadcast layout.
        gl.store(out_ptr + ((base + s[:, None] * M + r[None, :]) * C)[:, :, None] + c[None, None, :], acc)
        gl.store(ml_ptr + (base + s[:, None] * M + r[None, :]) * 2, m)
        cluster.barrier()
        s2 = gl.arange(0, NC, layout=gl.SliceLayout(1, gl.SliceLayout(2, LB)))
        r2 = gl.arange(0, M, layout=gl.SliceLayout(0, gl.SliceLayout(2, LB)))
        c2 = gl.arange(0, C, layout=gl.SliceLayout(0, gl.SliceLayout(1, LB)))
        acc = gl.load(out_ptr + ((base + s2[:, None] * M + r2[None, :]) * C)[:, :, None] + c2[None, None, :])
        m = gl.load(ml_ptr + (base + s2[:, None] * M + r2[None, :]) * 2)
        l = gl.load(ml_ptr + (base + s2[:, None] * M + r2[None, :]) * 2 + 1)
        mx = gl.max(m, 0)
        w = gl.exp2(m - mx[None, :])
        den = gl.sum(w * l, 0)
        o = gl.sum(w[:, :, None] * acc, 0) / gl.convert_layout(den, gl.SliceLayout(1, gl.SliceLayout(0, LB)))[:, None]
        rr = gl.arange(0, M, layout=gl.SliceLayout(1, gl.SliceLayout(0, LB)))
        cc = gl.arange(0, C, layout=gl.SliceLayout(0, gl.SliceLayout(0, LB)))
        gl.store(out_ptr + P * NC * M * C + (pid * M + rr[:, None]) * C + cc[None, :], o)
    elif MERGE:
        mx = gl.max(m, 0)
        w = gl.exp2(m - mx[None, :])
        den = gl.sum(w * l, 0)
        o = gl.sum(w[:, :, None] * acc, 0) / gl.convert_layout(den, gl.SliceLayout(1, gl.SliceLayout(0, L3)))[:, None]
        rr = gl.arange(0, M, layout=gl.SliceLayout(1, gl.SliceLayout(0, L3)))
        cc = gl.arange(0, C, layout=gl.SliceLayout(0, gl.SliceLayout(0, L3)))
        gl.store(out_ptr + (pid * M + rr[:, None]) * C + cc[None, :], o)
    else:
        gl.store(out_ptr + ((base + s[:, None] * M + r[None, :]) * C)[:, :, None] + c[None, None, :], acc)


def main():
    NC, M, C, P = int(sys.argv[1]), 16, 64, 2
    x = torch.randn(P * NC * M * C, device="cuda")
    ml = torch.randn(P * NC * M * 2, device="cuda").abs()
    out = torch.zeros(2 * P * NC * M * C, device="cuda")
    for merge in (0, 1, 2):
        k[(P,)](x, ml, out, NC=NC, M=M, C=C, MERGE=merge, P=P, LB=gl.BlockedLayout([1, 1, 4], [1, 2, 16], [1, 4, 1], [2, 1, 0], cga_layout=[[0, 0, 0]] * (NC.bit_length() - 1)), L3=gl.BlockedLayout([1, 1, 4], [1, 2, 16], [1, 4, 1], [2, 1, 0], cga_layout=[[1 << i, 0, 0] for i in range(NC.bit_length() - 1)]), num_ctas=NC, num_warps=4)
        torch.cuda.synchronize()
        if merge == 1:
            xs = x.view(P, NC, M, C); mls = ml.view(P, NC, M, 2)
            w = torch.exp2(mls[..., 0] - mls[..., 0].max(1, keepdim=True).values)
            ref = (w[..., None] * xs).sum(1) / (w * mls[..., 1]).sum(1)[..., None]
            print("err", (out[:P * M * C].view(P, M, C) - ref).abs().max().item())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(20):
                k[(P,)](x, ml, out, NC=NC, M=M, C=C, MERGE=merge, P=P, LB=gl.BlockedLayout([1, 1, 4], [1, 2, 16], [1, 4, 1], [2, 1, 0], cga_layout=[[0, 0, 0]] * (NC.bit_length() - 1)), L3=gl.BlockedLayout([1, 1, 4], [1, 2, 16], [1, 4, 1], [2, 1, 0], cga_layout=[[1 << i, 0, 0] for i in range(NC.bit_length() - 1)]), num_ctas=NC, num_warps=4)
        with open("/tmp/okc-gpu.lock", "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            g.replay(); torch.cuda.synchronize()
            ts = []
            for _ in range(20):
                a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                a.record(); g.replay(); b.record(); torch.cuda.synchronize()
                ts.append(a.elapsed_time(b) / 20 * 1000)
        print(NC, ["copy", "dsmem", "l2x"][merge], f"{sorted(ts)[10]:.2f} us/launch (warm, back to back)")


main()

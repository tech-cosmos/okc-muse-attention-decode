# Feasibility: Triton kernel (num_ctas=1 compile) launched as clusters of NC CTAs; each CTA pushes a [R, 64] fp32
# block into rank 0's smem (global_smem + OFF) with mapa + st.shared::cluster, cluster barrier, rank 0 sums.
import torch, triton, triton.language as tl

@triton.jit
def _cbar():
    tl.inline_asm_elementwise("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;\n\tmov.u32 $0, 0;",
                              "=r", [], dtype=tl.int32, is_pure=False, pack=1)

@triton.jit
def k(x_ptr, o_ptr, NC: tl.constexpr, R: tl.constexpr, OFF: tl.constexpr, STAGE: tl.constexpr):
    pid = tl.program_id(0)
    cl = pid // NC
    rank = tl.inline_asm_elementwise("mov.u32 $0, %cluster_ctarank;", "=r", [], dtype=tl.int32, is_pure=True, pack=1)
    rows = tl.arange(0, R)[:, None]; cols = tl.arange(0, 64)[None, :]
    x = tl.load(x_ptr + pid * R * 64 + rows * 64 + cols)
    # make Triton allocate smem (so global_smem exists and is big enough): a dummy convert via dot-free path
    if STAGE:
        w = tl.load(x_ptr + tl.arange(0, 64)[:, None] * 64 + cols).to(tl.float16)
        x = x + tl.dot(x.to(tl.float16), w) * 0.0
    _cbar()  # everyone done with own smem use
    off = (rank * R * 64 + rows * 64 + cols) * 4 + OFF
    tl.inline_asm_elementwise(
        "{ .reg .u32 a; .reg .u32 ra; mov.u32 a, global_smem; add.u32 a, a, $1; mapa.shared::cluster.u32 ra, a, 0;"
        " st.shared::cluster.f32 [ra], $2; mov.u32 $0, 0; }",
        "=r,r,r", [off.to(tl.int32), x], dtype=tl.int32, is_pure=False, pack=1)
    _cbar()
    if rank == 0:
        acc = tl.zeros([R, 64], tl.float32)
        for i in tl.static_range(NC):
            off2 = (i * R * 64 + rows * 64 + cols) * 4 + OFF
            acc += tl.inline_asm_elementwise(
                "{ .reg .u32 a; mov.u32 a, global_smem; add.u32 a, a, $1; ld.shared.f32 $0, [a]; }",
                "=r,r", [off2.to(tl.int32)], dtype=tl.float32, is_pure=False, pack=1)
        tl.store(o_ptr + cl * R * 64 + rows * 64 + cols, acc)

import sys; NC, R, ncl = int(sys.argv[1]), 16, 4
x = torch.randn(ncl * NC, R, 64, device="cuda"); o = torch.zeros(ncl, R, 64, device="cuda")
ck = k.warmup(x, o, NC=NC, R=R, OFF=0, STAGE=True, grid=(ncl * NC,), num_warps=4)
ck._init_handles()
print("shared", ck.metadata.shared, "regs", ck.n_regs)
print([l for l in ck.asm["ptx"].splitlines() if "global_smem" in l][:3])
meta = (ck.packed_metadata[0], NC, max(ck.packed_metadata[2], NC * R * 64 * 4))
stream = torch.cuda.current_stream().cuda_stream
ck.run(ncl, 1, 1, stream, ck.function, meta, None, None, None, x, o, NC, R, 0, True)
torch.cuda.synchronize()
ref = x.view(ncl, NC, R, 64).sum(1)
print("err", (o - ref).abs().max().item())

# Probe: Gluon mma_v2 with rank-3 (warp-batched) operands, FP8 k_width 4 and FP16, smem rank-3 rings.
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2, async_copy as cp

@gluon.jit
def k(q_ptr, k_ptr, v_ptr, o_ptr, W: gl.constexpr, M: gl.constexpr, BN: gl.constexpr, C: gl.constexpr):
    MMA: gl.constexpr = gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[W, 1, 1], instr_shape=[1, 16, 8])
    QA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=MMA, k_width=4)
    KB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=MMA, k_width=4)
    PA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=MMA, k_width=2)
    VB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=MMA, k_width=2)
    BL: gl.constexpr = gl.BlockedLayout([1, 1, 16], [1, 4, 8], [W, 1, 1], [2, 1, 0])
    SH: gl.constexpr = gl.SwizzledSharedLayout(16, 1, 8, [2, 1, 0])
    w = gl.arange(0, W, layout=gl.SliceLayout(1, gl.SliceLayout(2, BL)))
    n = gl.arange(0, BN, layout=gl.SliceLayout(0, gl.SliceLayout(2, BL)))
    c = gl.arange(0, C, layout=gl.SliceLayout(0, gl.SliceLayout(1, BL)))
    off = (w[:, None, None] * BN + n[None, :, None]) * C + c[None, None, :]
    ring = gl.allocate_shared_memory(gl.float8e4nv, [2, W, BN, C], SH)
    vring = gl.allocate_shared_memory(gl.float8e4nv, [2, W, BN, C], SH)
    cp.async_copy_global_to_shared(ring.index(1), k_ptr + off)
    cp.async_copy_global_to_shared(vring.index(1), v_ptr + off)
    cp.commit_group()
    m = gl.arange(0, M, layout=gl.SliceLayout(0, gl.SliceLayout(2, BL)))
    qo = (w[:, None, None] * M + m[None, :, None]) * C + c[None, None, :]
    q = gl.convert_layout(gl.load(q_ptr + qo), QA)
    cp.wait_group(0)
    gl.barrier()
    sc = mma_v2(q, ring.index(1).permute((0, 2, 1)).load(KB), gl.zeros([W, M, BN], gl.float32, MMA))
    mx = gl.max(sc, 2)
    p = gl.exp2(sc - mx[:, :, None])
    v = gl.convert_layout(vring.index(1).load(BL).to(gl.float16), VB)
    acc = mma_v2(gl.convert_layout(p.to(gl.float16), PA), v, gl.zeros([W, M, C], gl.float32, MMA))
    tot = gl.sum(acc, 0)  # cross-warp
    ro = gl.arange(0, M, layout=gl.SliceLayout(1, gl.SliceLayout(0, MMA)))
    co = gl.arange(0, C, layout=gl.SliceLayout(0, gl.SliceLayout(0, MMA)))
    gl.store(o_ptr + ro[:, None] * C + co[None, :], tot)

W, M, BN, C = 4, 16, 32, 128
torch.manual_seed(0)
q = (torch.randn(W, M, C, device="cuda") * 0.5).to(torch.float8_e4m3fn)
kk = (torch.randn(W, BN, C, device="cuda") * 0.5).to(torch.float8_e4m3fn)
vv = (torch.randn(W, BN, C, device="cuda")).to(torch.float8_e4m3fn)
o = torch.empty(M, C, device="cuda")
k[(1,)](q, kk, vv, o, W, M, BN, C, num_warps=W)
s = q.float() @ kk.float().transpose(1, 2)
p = torch.exp2(s - s.amax(2, keepdim=True)).half().float()
ref = (p @ vv.float()).sum(0)
print("max err", (o - ref).abs().max().item(), "ref max", ref.abs().max().item())

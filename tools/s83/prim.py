"""s83: primitives for the tensor-core V transpose (TV): VT = I @ V^T as FP8 wgmma with an f16 accumulator, used
as the register A operand of O^T = VT @ P^T (P FP16 stored to smem)."""
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import fence_async_shared, warpgroup_mma, warpgroup_mma_wait

@gluon.jit
def k(v_ptr, p_ptr, o_ptr, M: gl.constexpr, D: gl.constexpr, BN: gl.constexpr, ACC16: gl.constexpr):
    BL: gl.constexpr = gl.BlockedLayout([1, 8], [4, 8], [4, 1], [1, 0])
    SH8: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=8, rank=2)
    SQ: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=64, element_bitwidth=8, rank=2)
    SH16: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=16, rank=2)
    T_L: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, BN, 32])
    OT_L: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, M, 16])
    A_L: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=OT_L, k_width=2)
    rn = gl.arange(0, BN, layout=gl.SliceLayout(1, BL))
    cd = gl.arange(0, D, layout=gl.SliceLayout(0, BL))
    v = gl.load(v_ptr + rn[:, None] * D + cd[None, :])           # fp8 [BN, D]
    v_s = gl.allocate_shared_memory(gl.float8e4nv, [BN, D], SQ, v)
    rd = gl.arange(0, D, layout=gl.SliceLayout(1, BL))
    ident = gl.where(rd[:, None] == cd[None, :], 1.0, 0.0).to(gl.float8e4nv)
    i_s = gl.allocate_shared_memory(gl.float8e4nv, [D, D], SQ, ident)
    rm = gl.arange(0, M, layout=gl.SliceLayout(1, BL))
    cn = gl.arange(0, BN, layout=gl.SliceLayout(0, BL))
    p = gl.load(p_ptr + rm[:, None] * BN + cn[None, :])           # f16 [M, BN]
    p_s = gl.allocate_shared_memory(gl.float16, [M, BN], SH16, p)
    fence_async_shared()
    gl.barrier()
    if ACC16:
        vt = warpgroup_mma(i_s, v_s.permute((1, 0)), gl.zeros([D, BN], gl.float16, T_L), use_acc=False)
    else:
        vt = warpgroup_mma(i_s, v_s.permute((1, 0)), gl.zeros([D, BN], gl.float32, T_L), use_acc=False).to(gl.float16)
    a = gl.convert_layout(vt, A_L)
    ot = warpgroup_mma(a, p_s.permute((1, 0)), gl.zeros([D, M], gl.float32, OT_L))
    ro = gl.arange(0, D, layout=gl.SliceLayout(1, OT_L))
    co = gl.arange(0, M, layout=gl.SliceLayout(0, OT_L))
    gl.store(o_ptr + co[None, :] * D + ro[:, None], ot)       # O[row][d]

for acc16 in (True, False):
    M, D, BN = 64, 64, 64
    v = (torch.randn(BN, D, device="cuda") * 3).to(torch.float8_e4m3fn)
    p = torch.rand(M, BN, device="cuda").half()
    o = torch.empty(M, D, device="cuda")
    k[(1,)](v, p, o, M, D, BN, acc16, num_warps=4)
    ref = p.float() @ v.float()
    print("acc16", acc16, "max err", (o - ref).abs().max().item(), "ref max", ref.abs().max().item())

"""Unit test: FP8 wgmma with the A operand (queries) in registers vs in shared memory (Gluon, sm_90)."""
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import warpgroup_mma, warpgroup_mma_wait, fence_async_shared


@gluon.jit
def k(q_ptr, k_ptr, o_ptr, QREG: gl.constexpr, KW: gl.constexpr):
    M: gl.constexpr = 64
    N: gl.constexpr = 64
    D: gl.constexpr = 64
    BL: gl.constexpr = gl.BlockedLayout([1, 16], [4, 8], [4, 1], [1, 0])
    S_L: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, N, 32])
    r = gl.arange(0, M, layout=gl.SliceLayout(1, BL))
    c = gl.arange(0, D, layout=gl.SliceLayout(0, BL))
    q = gl.load(q_ptr + r[:, None] * D + c[None, :])
    kk = gl.load(k_ptr + r[:, None] * D + c[None, :])
    SQ: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=64, element_bitwidth=8, rank=2)
    SK: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=64, element_bitwidth=8, rank=2)
    q_s = gl.allocate_shared_memory(gl.float8e4nv, [M, D], SQ, q)
    k_s = gl.allocate_shared_memory(gl.float8e4nv, [N, D], SK, kk)
    fence_async_shared()
    gl.barrier()
    if QREG:
        qa = gl.convert_layout(q_s.load(BL), gl.DotOperandLayout(operand_index=0, parent=S_L, k_width=KW))
    else:
        qa = q_s
    t = warpgroup_mma(qa, k_s.permute((1, 0)), gl.zeros([M, N], gl.float32, S_L), use_acc=False, is_async=False)
    sc = warpgroup_mma_wait(0, deps=[t])
    ro = gl.arange(0, M, layout=gl.SliceLayout(1, S_L))
    co = gl.arange(0, N, layout=gl.SliceLayout(0, S_L))
    gl.store(o_ptr + ro[:, None] * N + co[None, :], sc)


q = torch.randn(64, 64, device="cuda").to(torch.float8_e4m3fn)
kk = torch.randn(64, 64, device="cuda").to(torch.float8_e4m3fn)
ref = q.float() @ kk.float().T
for qreg, kw in ((False, 4), (True, 4), (True, 2), (True, 8), (True, 16)):
    o = torch.empty(64, 64, device="cuda")
    try:
        k[(1,)](q, kk, o, QREG=qreg, KW=kw, num_warps=4)
        print(qreg, kw, (o - ref).abs().max().item())
    except Exception as e:
        print(qreg, kw, "ERR", str(e).split("\n")[0][:150])

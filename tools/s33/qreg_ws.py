"""FP8 register-A wgmma inside warp_specialize partitions (two consumer partitions + a dummy worker)."""
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import warpgroup_mma, warpgroup_mma_wait, fence_async_shared


@gluon.jit
def _part(q_s, k_s, o_ptr, HEAD: gl.constexpr, QREG: gl.constexpr):
    M: gl.constexpr = 64
    N: gl.constexpr = 64
    S_L: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, N, 32])
    BL: gl.constexpr = gl.BlockedLayout([1, 16], [4, 8], [4, 1], [1, 0])
    if QREG:
        qa = gl.convert_layout(q_s.index(HEAD).load(BL), gl.DotOperandLayout(operand_index=0, parent=S_L, k_width=4))
    else:
        qa = q_s.index(HEAD)
    t = warpgroup_mma(qa, k_s.slice(HEAD * 64, 64, 1).permute((1, 0)), gl.zeros([M, N], gl.float32, S_L), use_acc=False, is_async=False)
    sc = warpgroup_mma_wait(0, deps=[t])
    ro = gl.arange(0, M, layout=gl.SliceLayout(1, S_L))
    co = gl.arange(0, N, layout=gl.SliceLayout(0, S_L))
    gl.store(o_ptr + HEAD * M * N + ro[:, None] * N + co[None, :], sc)


@gluon.jit
def _idle(x):
    pass


@gluon.jit
def k(q_ptr, k_ptr, o_ptr, QREG: gl.constexpr):
    BL: gl.constexpr = gl.BlockedLayout([1, 16], [4, 8], [4, 1], [1, 0])
    r = gl.arange(0, 64, layout=gl.SliceLayout(1, BL))
    c = gl.arange(0, 64, layout=gl.SliceLayout(0, BL))
    SQ: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=64, element_bitwidth=8, rank=2)
    SK: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=128, element_bitwidth=8, rank=2)
    c2 = gl.arange(0, 128, layout=gl.SliceLayout(0, BL))
    q_s = gl.allocate_shared_memory(gl.float8e4nv, [2, 64, 64], SQ)
    q_s.index(0).store(gl.load(q_ptr + r[:, None] * 64 + c[None, :]))
    q_s.index(1).store(gl.load(q_ptr + 4096 + r[:, None] * 64 + c[None, :]))
    k_s = gl.allocate_shared_memory(gl.float8e4nv, [64, 128], SK, gl.load(k_ptr + r[:, None] * 128 + c2[None, :]))
    fence_async_shared()
    gl.barrier()
    gl.warp_specialize([(_part, (q_s, k_s, o_ptr, 0, QREG)), (_part, (q_s, k_s, o_ptr, 1, QREG)),
                        (_idle, (q_ptr,))], [4, 4], [104, 32])


q = torch.randn(2, 64, 64, device="cuda").to(torch.float8_e4m3fn)
kk = torch.randn(64, 128, device="cuda").to(torch.float8_e4m3fn)
ref = torch.stack([q[h].float() @ kk[:, h * 64:(h + 1) * 64].float().T for h in range(2)])
for qreg in (False, True):
    o = torch.full((2, 64, 64), float("nan"), device="cuda")
    k[(1,)](q, kk, o, QREG=qreg, num_warps=4, maxnreg=80)
    print(qreg, (o - ref).abs().max().item())

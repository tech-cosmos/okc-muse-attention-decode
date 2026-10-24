"""FP8 register-A wgmma: A made by .to(fp8) from fp32 (a) in the dot layout, (b) in a blocked layout + convert,
(c) fp8 loaded in the dot layout."""
import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import warpgroup_mma, warpgroup_mma_wait, fence_async_shared


@gluon.jit
def k(x_ptr, q8_ptr, k_ptr, o_ptr, MODE: gl.constexpr):
    M: gl.constexpr = 64
    N: gl.constexpr = 64
    D: gl.constexpr = 64
    BL: gl.constexpr = gl.BlockedLayout([1, 16], [4, 8], [4, 1], [1, 0])
    S_L: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, N, 32])
    QD: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=S_L, k_width=4)
    r = gl.arange(0, M, layout=gl.SliceLayout(1, BL))
    c = gl.arange(0, D, layout=gl.SliceLayout(0, BL))
    SK: gl.constexpr = gl.NVMMASharedLayout(swizzle_byte_width=64, element_bitwidth=8, rank=2)
    k_s = gl.allocate_shared_memory(gl.float8e4nv, [N, D], SK, gl.load(k_ptr + r[:, None] * D + c[None, :]))
    fence_async_shared()
    gl.barrier()
    rd = gl.arange(0, M, layout=gl.SliceLayout(1, QD))
    cd = gl.arange(0, D, layout=gl.SliceLayout(0, QD))
    if MODE == 0:
        qa = gl.load(x_ptr + rd[:, None] * D + cd[None, :]).to(gl.float8e4nv)
    elif MODE == 1:
        qa = gl.convert_layout(gl.load(x_ptr + r[:, None] * D + c[None, :]).to(gl.float8e4nv), QD)
    else:
        qa = gl.load(q8_ptr + rd[:, None] * D + cd[None, :])
    t = warpgroup_mma(qa, k_s.permute((1, 0)), gl.zeros([M, N], gl.float32, S_L), use_acc=False, is_async=False)
    sc = warpgroup_mma_wait(0, deps=[t])
    ro = gl.arange(0, M, layout=gl.SliceLayout(1, S_L))
    co = gl.arange(0, N, layout=gl.SliceLayout(0, S_L))
    gl.store(o_ptr + ro[:, None] * N + co[None, :], sc)


x = torch.randn(64, 64, device="cuda")
q8 = x.to(torch.float8_e4m3fn)
kk = torch.randn(64, 64, device="cuda").to(torch.float8_e4m3fn)
ref = q8.float() @ kk.float().T
for mode in (0, 1, 2):
    o = torch.empty(64, 64, device="cuda")
    k[(1,)](x, q8, kk, o, MODE=mode, num_warps=4)
    print(mode, (o - ref).abs().max().item())

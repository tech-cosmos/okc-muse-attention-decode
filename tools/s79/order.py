# Which (row, col) of the S_L ([64, 64], wgmma v3 [16,64,32], 4 warps) layout is pack slot j / lane of a pack-32 asm?
import torch, triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
ASM = "{\n.reg .u32 t;\nmov.u32 t, %laneid;\n" + "".join(f"mad.lo.u32 ${j}, t, 100, {j};\n" for j in range(32)) + "}"
CON = ",".join(["=r"] * 32 + ["r"] * 32)
@gluon.jit
def k(out, ASM: gl.constexpr, CON: gl.constexpr):
    S_L: gl.constexpr = gl.NVMMADistributedLayout(version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 64, 32])
    r = gl.arange(0, 64, layout=gl.SliceLayout(1, S_L))
    c = gl.arange(0, 64, layout=gl.SliceLayout(0, S_L))
    x = r[:, None] * 0 + c[None, :] * 0
    y = gl.inline_asm_elementwise(ASM, CON, [x], dtype=gl.int32, is_pure=True, pack=32)
    gl.store(out + r[:, None] * 64 + c[None, :], y)
o = torch.zeros(64, 64, dtype=torch.int32, device="cuda")
k[(1,)](o, ASM, CON, num_warps=4)
o = o.cpu()
for row in (0, 1, 8, 9, 16):
    print(row, [(int(v) // 100, int(v) % 100) for v in o[row, :20]])

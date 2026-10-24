import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
@gluon.jit
def k(out_ptr, NC: gl.constexpr):
    L: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0], cga_layout=[[1], [2]])
    i = gl.arange(0, 512, layout=L)
    pid = gl.program_id(0)
    rank = gl.inline_asm_elementwise("mov.u32 $0, %cluster_ctarank;", "=r,r", [i], dtype=gl.int32, is_pure=True, pack=1)
    gl.store(out_ptr + pid * 512 + i, rank * 1000 + pid)
out = torch.full((8 * 512,), -1, dtype=torch.int32, device="cuda")
k[(8,)](out, NC=4, num_ctas=4)
torch.cuda.synchronize()
print(out.view(8, 4, 128)[:, :, 0])

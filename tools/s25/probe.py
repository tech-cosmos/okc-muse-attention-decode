import torch, triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.hopper import cluster

@gluon.jit
def k(out_ptr, NC: gl.constexpr):
    L: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0], cga_layout=[[0]] * (NC.bit_length() - 1))
    i = gl.arange(0, 128, layout=L)
    pid = gl.program_id(0)
    rank = gl.inline_asm_elementwise("mov.u32 $0, %cluster_ctarank;", "=r,r", [i], dtype=gl.int32, is_pure=True, pack=1)
    cluster.barrier()
    gl.store(out_ptr + (pid * NC + rank) * 128 + i, rank * 1000 + pid)

out = torch.full((8 * 4 * 128,), -1, dtype=torch.int32, device="cuda")
k[(8,)](out, NC=4, num_ctas=4)
torch.cuda.synchronize()
print(out.view(8, 4, 128)[:, :, 0])

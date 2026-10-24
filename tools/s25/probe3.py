import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
@gluon.jit
def k(x_ptr, out_ptr):
    L: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0], cga_layout=((1, 0), (2, 0)))
    r = gl.arange(0, 4, layout=gl.SliceLayout(1, L))
    c = gl.arange(0, 128, layout=gl.SliceLayout(0, L))
    x = gl.load(x_ptr + r[:, None] * 128 + c[None, :])
    m = gl.max(x, 0)
    s = gl.sum(gl.exp(x - m[None, :]), 0)
    gl.store(out_ptr + c, s)
x = torch.randn(4, 128, device="cuda")
out = torch.zeros(128, device="cuda")
k[(1,)](x, out, num_ctas=4)
torch.cuda.synchronize()
ref = torch.exp(x - x.max(0).values).sum(0)
print((out - ref).abs().max())
if __name__ != "__main__":
    kk = k.warmup(x, out, num_ctas=4, grid=(1,))
    ptx = kk.asm["ptx"]
    import re
    for pat in ["mapa", "shared::cluster", "barrier.cluster", "fence", "bar.sync"]:
        print(pat, len(re.findall(re.escape(pat), ptx)))

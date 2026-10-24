import torch, triton, triton.language as tl
@triton.jit
def k(x_ptr, o_ptr, r_ptr, N: tl.constexpr):
    pid = tl.program_id(0)
    rows = tl.arange(0, 16)[:, None]; cols = tl.arange(0, N)[None, :]
    x = tl.load(x_ptr + pid * 16 * N + rows * N + cols)
    s = tl.sum(x, axis=1)
    tl.store(o_ptr + pid * 16 + tl.arange(0, 16), s)
    rk = tl.inline_asm_elementwise("mov.u32 $0, %cluster_ctarank;", "=r", [], dtype=tl.int32, is_pure=True, pack=1)
    tl.store(r_ptr + pid, rk)
x = torch.randn(4, 16, 1024, device="cuda"); o = torch.empty(4, 16, device="cuda"); r = torch.zeros(4, dtype=torch.int32, device="cuda")
h = k[(4,)](x, o, r, N=1024, num_ctas=8, num_warps=4)
torch.cuda.synchronize()
print((o - x.sum(-1)).abs().max().item(), r.tolist())
ir = h.asm["ttgir"]
import re; print([l for l in ir.splitlines() if "CTAsPerCGA" in l or "cga" in l.lower()][:5])

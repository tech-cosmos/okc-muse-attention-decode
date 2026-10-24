"""Per-row bulk-copy gather ceiling (tools/s74/tgather.cu) vs tools/gather.py's 236 us (H100, B 32 ctx 80K)."""
import fcntl, statistics, torch
from torch.utils.cpp_extension import load_inline
src = open("/home/daytona/repo/tools/s74/tgather.cu").read()
m = load_inline(name="s74_tgather2", cpp_sources="void tgather(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);",
                cuda_sources=src, functions=["tgather"], extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"])
N = 32 * 80000
kc = torch.randint(0, 255, (N, 128), dtype=torch.uint8, device="cuda")
vc = torch.randint(0, 255, (N, 128), dtype=torch.uint8, device="cuda")
idx = torch.randperm(N, device="cuda").int()
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
lock = open("/tmp/okc-gpu.lock", "w")
for chunk in (640, 1280, 2560):
    for bn, ns in [(32, 4), (32, 8), (64, 4)]:
        if chunk % bn: continue
        out = torch.empty(N // chunk, device="cuda")
        f = lambda: m.tgather(kc, vc, idx, out, chunk, bn, ns)
        f(); torch.cuda.synchronize(); ts = []
        fcntl.flock(lock, fcntl.LOCK_EX)
        for _ in range(11):
            flush.zero_(); a = torch.cuda.Event(True); b = torch.cuda.Event(True)
            a.record(); f(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1000)
        fcntl.flock(lock, fcntl.LOCK_UN)
        t = statistics.median(ts)
        print(f"chunk {chunk:6d} ctas {N // chunk:5d} bn {bn:3d} ns {ns}: {t:7.1f} us {2 * N * 128 / t / 1e6:.2f} TB/s", flush=True)

"""Fixed costs under graph replay with a cold L2 (H100): truly empty graph vs one tiny kernel, and flush by
write (bench.py) vs by read (the judge's description)."""
import fcntl, statistics, sys, torch
sys.path.insert(0, ".")
FL = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
FL64 = FL.view(torch.int64)
def tg(fn, flush, reps=100):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph(); s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    ts = []
    with open("/tmp/okc-gpu.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for _ in range(reps):
            flush()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); g.replay(); b.record(); torch.cuda.synchronize()
            ts.append(a.elapsed_time(b) * 1000)
    return statistics.median(ts)
x = torch.zeros(1, device="cuda")
sink = torch.zeros((), dtype=torch.int64, device="cuda")
flushes = {"write": lambda: FL.zero_(), "read": lambda: torch.sum(FL64, 0, out=sink)}
import reference, submission
shapes = {"decode 1 sliding": dict(batch=1, ctx=80000, heads="tp8", layer="sliding", page=1, q_len=1, dist="soft"),
          "verify 1 sliding": dict(batch=1, ctx=80000, heads="tp8", layer="sliding", page=1, q_len=8, dist="soft"),
          "decode 1 full": dict(batch=1, ctx=80000, heads="tp8", layer="full", page=1, q_len=1, dist="coherent")}
if __name__ == "__main__":
 for fname, fl in flushes.items():
     e0 = tg(lambda: None, fl); e1 = tg(lambda: x.add_(1), fl); e2 = tg(lambda: (x.add_(1), x.add_(1)), fl)
     print(f"flush {fname}: empty graph {e0:.2f}  1 tiny kernel {e1:.2f}  2 tiny {e2:.2f}")
     for n, c in shapes.items():
         d = reference.generate_input(seed=99, **c)
         t = tg(lambda: submission.kernel_fn(d), fl)
         print(f"   {n}: raw {t:.2f}  -empty {t-e0:.2f}  -1kernel {t-e1:.2f}")

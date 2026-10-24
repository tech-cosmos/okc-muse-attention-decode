"""H200 emulation on the H100: park a spinner on BLOCK SMs so the remaining SMs see the H200's per-SM HBM
bandwidth (H100 3.35 TB/s over 92 SMs ~= H200 4.8 TB/s over 132), plan for the free SMs, and time CUDA-graph
replays (cold L2, median, minus an empty replay) while the spinner holds its SMs.

  python tools/emu.py [block=40] [shape filter, e.g. 'full'] ['{"plan": "overrides"}']
"""
import fcntl
import json
import os
import statistics
import sys

os.environ["CUDA_MODULE_LOADING"] = "EAGER"  # a lazily loaded kernel would wait for the spinner
import torch
from torch.utils.cpp_extension import load_inline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("PATH", "")
os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ["PATH"]

import bench  # noqa: E402
import submission  # noqa: E402

_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
__global__ void spin(volatile int* flag) {
    extern __shared__ char sm[];
    if (threadIdx.x == 0) sm[0] = 0;
    while (*flag == 0) { __nanosleep(2000); }
}
void launch(torch::Tensor flag, int64_t blocks, int64_t stream_ptr) {
    int smem = 200 * 1024;
    cudaFuncSetAttribute(spin, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    spin<<<blocks, 1024, smem, (cudaStream_t)stream_ptr>>>((volatile int*)flag.data_ptr<int>());
}
"""
_CPP = "void launch(torch::Tensor flag, int64_t blocks, int64_t stream_ptr);"
_mod = None


def _spinner():
    global _mod
    if _mod is None:
        _mod = load_inline("emu_spin", cpp_sources=_CPP, cuda_sources=_SRC, functions=["launch"],
                           build_directory=_build_dir(), verbose=False)
    return _mod


def _build_dir():
    d = "/tmp/emu_build"
    os.makedirs(d, exist_ok=True)
    return d


_FLUSH = None


def time_blocked(fn, block, reps=30):
    """Median replay time (us) of fn's CUDA graph while `block` SMs are held by a spinner."""
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
        with torch.cuda.graph(g, stream=s):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    flag = torch.zeros(1, dtype=torch.int32, device="cuda")
    # Every kernel used while the spinner runs must be loaded before it starts.
    flag.fill_(0)
    _FLUSH.zero_()
    torch.cuda._sleep(1000)
    g.replay()
    torch.cuda.synchronize()
    spin_stream = torch.cuda.Stream()
    work = torch.cuda.Stream()
    times = []
    with open("/tmp/okc-gpu.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if block:
            _spinner().launch(flag, block, spin_stream.cuda_stream)
        with torch.cuda.stream(work):
            torch.cuda._sleep(2_000_000)  # let the spinner take its SMs
            for i in range(reps + 3):
                _FLUSH.zero_()
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                g.replay()
                b.record()
                b.synchronize()
                if i >= 3:
                    times.append(a.elapsed_time(b) * 1000)
            flag.fill_(1)
        torch.cuda.synchronize()
    return statistics.median(times)


def run(block, filt="", plan=None, reps=30):
    sms = 132 - block
    submission._SMS[0] = sms
    x = torch.zeros(1, device="cuda")
    empty = time_blocked(lambda: x.add_(1), block, reps=60)
    orig = submission._plan
    res = {}
    try:
        if plan:
            submission._plan = lambda cfg, mp, dev: {**orig(cfg, mp, dev), **plan}
        for load, c in bench.SCORED:
            if filt and filt not in f"{load} {c['layer']}":
                continue
            data = bench.data_for(load, c)
            t = time_blocked(lambda: submission.kernel_fn(data), block, reps) - empty
            res[(load, c["layer"])] = t
            print(f"{load:10s} {c['layer']:8s} {t:8.2f} us  (sglang H200 {bench.SGLANG_H200[(load, c['layer'])]:7.2f})",
                  flush=True)
    finally:
        submission._plan = orig
        submission._SMS[0] = 132
    return res


if __name__ == "__main__":
    block = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    filt = sys.argv[2] if len(sys.argv) > 2 else ""
    plan = json.loads(sys.argv[3]) if len(sys.argv) > 3 else None
    run(block, filt, plan)

import sys, torch, importlib, statistics, fcntl
sys.path.insert(0, "/home/daytona/repo2/tools"); sys.path.insert(0, "/tmp/a2p")
import bench
base = open("/home/daytona/repo2/submission.py").read()
def mk(pair):
    full = base.replace("#define NV_PAIR 1", f"#define NV_PAIR {pair}")
    a0 = full.index("_NDS_SRC = r'''"); a1 = full.index("NDS_MIN_BATCH = 32")
    s = full[a0:a1]
    def rep(a, b):
        nonlocal s
        assert s.count(a) == 1, a; s = s.replace(a, b)
    T = '{ unsigned long long _t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(_t)); if (lane == 0) nds_st[VID * 4 + K] = _t; }'
    vid = "(blockIdx.x * (1 + NV_PAIR) + pv)"
    st = lambda k: T.replace("VID", vid).replace("K", str(k))
    rep("template <int NS> struct Pair {", "__device__ unsigned long long nds_st[4096];\ntemplate <int NS> struct Pair {")
    rep("    const uint32_t qa = smem_u32(qd);\n", "    const uint32_t qa = smem_u32(qd);\n    if (w == 0) " + st(0) + "\n")
    rep("    if (n > 0) {\n        mb_wait(&sm.ready[0], 0);\n", "    if (n > 0) {\n        mb_wait(&sm.ready[0], 0);\n        if (w == 0) " + st(1) + "\n")
    rep("    if (!live) return;\n    l[0] += __shfl_xor_sync", "    if (w == 0) " + st(2) + "\n    if (!live) return;\n    l[0] += __shfl_xor_sync")
    rep("void nds(torch::Tensor qkv,", "torch::Tensor nds_stamps() { auto t = torch::empty({4096}, torch::dtype(torch::kInt64)); cudaMemcpyFromSymbol(t.data_ptr(), nv::nds_st, 4096 * 8); return t; }\nvoid nds(torch::Tensor qkv,")
    rep('functions=["nds"]', 'functions=["nds", "nds_stamps"]')
    rep('"int64_t, int64_t, int64_t);",\n        cuda_sources=_NDS_SRC', '"int64_t, int64_t, int64_t); torch::Tensor nds_stamps();",\n        cuda_sources=_NDS_SRC')
    s = (full[:a0] + s + full[a1:]).replace('name="muse_nds_a2s16p"', f'name="muse_nds_st{pair}"')
    open(f"/tmp/a2p/st{pair}.py", "w").write(s)
    return importlib.import_module(f"st{pair}")
for pair in (0, 1):
    m = mk(pair)
    for load, c in bench.SCORED:
        if f"{load} {c['layer']}" != "decode 32 sliding": continue
        data = bench.data_for(load, c)
        res = []
        with open("/tmp/okc-gpu.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            for r in range(5):
                m.kernel_fn(data); torch.cuda.synchronize()
                t = m._NDS.nds_stamps().view(-1, 4)[:256].double()
                t0 = t[:, 0].min()
                res.append([(t[:, 0] - t0).median().item(), (t[:, 0] - t0).max().item(), (t[:, 1] - t[:, 0]).median().item(),
                            (t[:, 2] - t[:, 1]).median().item(), (t[:, 2] - t[:, 1]).max().item(), (t[:, 2] - t0).max().item()])
        print(f"pair {pair}: start med/max, first tile, loop med/max, end max (ns):", [round(statistics.median(x)) for x in zip(*res)])

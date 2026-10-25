"""s121: nvs variants (#define overrides) at verify 1 full, paired cold vs main. usage: var.py '<json list of
{"defs": {"NV_NS": 8, ...}, "S": 64}>' [rounds]"""
import sys, json, statistics, hashlib; sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench, submission, re
vs, rounds = json.loads(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 7
mods = []
for v in vs:
    src = submission._NVS_SRC
    for k, x in v.get("defs", {}).items():
        src, n = re.subn(rf"#define {k} \S+", f"#define {k} {x}", src, count=1)
        assert n == 1, k
    name = "s121nvs_" + hashlib.md5(src.encode()).hexdigest()[:8]
    mods.append(submission._load_inline(
        name=name, cpp_sources="void nvs(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, "
        "torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, int64_t, "
        "int64_t, int64_t);", cuda_sources=src, functions=["nvs"],
        extra_cuda_cflags=["-gencode=arch=compute_90a,code=sm_90a", "-O3"], extra_cflags=["-O3"], verbose=False))
load, c = [x for x in bench.SCORED if f"{x[0]} {x[1]['layer']}" == "verify 1 full"][0]
res = [[] for _ in range(len(vs) + 1)]
for r in range(rounds):
    submission.NVS_FULL1 = False
    res[0].append(bench.run_shape(load, c, None))
    for i, (v, m) in enumerate(zip(vs, mods)):
        submission.NVS_FULL1, submission.NVS_F1_S, submission._NVS = True, v.get("S", 64), m
        res[i + 1].append(bench.run_shape(load, c, None))
for v, t in zip([{"main": 1}] + vs, res):
    print(f"  {statistics.median(t):8.2f} us  wins vs main {sum(a < b for a, b in zip(t, res[0]))}/{rounds}  {v}", flush=True)

import sys, re, statistics, importlib, os
cfgs = [tuple(map(int, x.split(","))) for x in sys.argv[1:]]  # NS,LAG
base = open("/tmp/sub_dbg.py").read()
res = {}
sys.path.insert(0, "/home/daytona/repo2/tools"); sys.path.insert(0, "/tmp")
import bench
for ns, lag in cfgs:
    src = base.replace("#define NV_NS 4", f"#define NV_NS {ns}").replace("#define NV_LAG 1", f"#define NV_LAG {lag}").replace('name="muse_nvs_a2dbg"', f'name="muse_nvs_a2lag{ns}_{lag}"')
    src = src.replace("NVS_MIN_BATCH = 32", "NVS_MIN_BATCH = 16")
    mn = f"a2sub_{ns}_{lag}"; open(f"/tmp/{mn}.py", "w").write(src)
    m = importlib.import_module(mn); bench.submission = m
    for l, c in bench.SCORED:
        if c["layer"] == "sliding" and l in ("verify 16", "verify 32"):
            t = [bench.run_shape(l, c) for _ in range(7)]
            print(f"NS {ns} LAG {lag} {l}: {statistics.median(t):.2f}", flush=True)

"""agent2 s5: paired A/B of nvs NV_C2 1 (one softmax chain per KV head) vs 2 (two chains, even/odd tiles) on verify
16/32 sliding. Usage: c2ab.py [rounds] [extra B define replacements as OLD=>NEW ...]"""
import sys, statistics, importlib
sys.path.insert(0, "/home/daytona/repo2/tools"); sys.path.insert(0, "/tmp")
import bench
base = open("/home/daytona/repo2/submission.py").read().replace("NVS_MIN_BATCH = 32", "NVS_MIN_BATCH = 16")
rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 9
extra = [a.split("=>") for a in sys.argv[2:]]
def mk(tag, c2, rep):
    src = base.replace("#define NV_C2 2", f"#define NV_C2 {c2}").replace('name="muse_nvs_a2s5b"', f'name="muse_nvs_a2s5b{tag}"')
    for a, b in rep:
        assert a in src, a
        src = src.replace(a, b)
    open(f"/tmp/a2c2_{tag}.py", "w").write(src)
    return importlib.import_module(f"a2c2_{tag}")
tag = "".join(c for c in "_".join(b for _, b in extra) if c.isalnum())[:20]
A = mk("A", 1, []); B = mk("B" + tag, 2, extra)
for l, c in bench.SCORED:
    if c["layer"] == "sliding" and l in ("verify 16", "verify 32"):
        d, ta, tb = [], [], []
        for r in range(rounds):
            t = {}
            for m in ((A, B) if r % 2 == 0 else (B, A)):
                bench.submission = m; t[m] = bench.run_shape(l, c)
            ta.append(t[A]); tb.append(t[B]); d.append(t[B] - t[A])
        print(f"{l}: C2=1 {statistics.median(ta):.2f} C2=2 {statistics.median(tb):.2f} paired {statistics.median(d):+.3f}, B wins {sum(x < 0 for x in d)}/{rounds}", flush=True)

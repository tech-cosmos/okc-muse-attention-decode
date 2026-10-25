"""Paired A/B of submission.py variants on the batched sliding shapes (H100, bench.run_shape).
  python tools/a2/vab.py ROUNDS "verify 32,verify 16" 'name|old=>new|old=>new' ...
Variant 0 is the unmodified submission (with NVS_MIN_BATCH 16 if 'nvs16' is given as a variant name prefix)."""
import sys, os, statistics, importlib, hashlib
sys.path.insert(0, "/home/daytona/repo2/tools"); sys.path.insert(0, "/tmp/a2v")
import bench
rounds = int(sys.argv[1]); loads = sys.argv[2].split(",")
base = open(os.environ.get("VAB_BASE", "/home/daytona/repo2/submission.py")).read()
mods = []
for spec in ["base"] + sys.argv[3:]:
    parts = spec.split("|"); name = parts[0]; src = base
    for p in parts[1:]:
        old, new = p.split("=>")
        assert old in src, f"{name}: {old!r} not found"
        src = src.replace(old, new)
    h = hashlib.md5(src.encode()).hexdigest()[:8]
    src = src.replace('name="muse_nvs_a2s9"', f'name="muse_nvs_v{h}"').replace('name="muse_csl', f'name="muse_csl{h}').replace('name="muse_nds', f'name="muse_nds{h}')
    mn = f"v_{h}"; open(f"/tmp/a2v/{mn}.py", "w").write(src)
    mods.append((name, importlib.import_module(mn)))
for l, c in bench.SCORED:
    if c["layer"] != "sliding" or l not in loads: continue
    t = {n: [] for n, _ in mods}
    for r in range(rounds):
        order = mods if r % 2 == 0 else mods[::-1]
        for n, m in order:
            bench.submission = m; t[n].append(bench.run_shape(l, c))
    b = t["base"]
    s = f"{l}: base {statistics.median(b):.2f}"
    for n, _ in mods[1:]:
        d = [x - y for x, y in zip(t[n], b)]
        s += f" | {n} {statistics.median(t[n]):.2f} ({statistics.median(d):+.2f}, {sum(x < 0 for x in d)}/{rounds})"
    print(s, flush=True)

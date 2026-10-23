"""Timing-only ablations of the main tile on the verify full layers (results are wrong on purpose)."""
import importlib, sys, types
sys.path.insert(0, ".")
import bench
src = open("submission.py").read()
TILE = '''    m_new = tl.maximum(m_i, tl.max(s, 1))
    alpha = tl.exp2(m_i - m_new)
    p = tl.exp2(s - m_new[:, None])
    l_i = l_i * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None] + tl.dot(p.to(DOT), v.to(DOT))
    return m_new, l_i, acc'''
assert src.count(TILE) == 1
VARIANTS = {
    "full": TILE,
    "no exp (p = s - m)": TILE.replace("p = tl.exp2(s - m_new[:, None])", "p = s - m_new[:, None]"),
    "no PV dot": TILE.replace("tl.dot(p.to(DOT), v.to(DOT))", "tl.sum(p, 1)[:, None] + v.to(tl.float32).sum(0)[None, :].to(tl.float32)"),
    "no V convert (dot on raw bytes)": TILE.replace("v.to(DOT)", "v.to(tl.uint8, bitcast=True).to(DOT)"),
    "no softmax at all (acc += s @ v)": '''    acc = acc + tl.dot(s.to(DOT), v.to(DOT))
    return m_i, l_i, acc''',
}
for i, (name, body) in enumerate(VARIANTS.items()):
    # Triton reads kernel source from disk, so each variant is a real module file.
    open(f"variant{i}.py", "w").write(src.replace(TILE, body))
    bench.submission = importlib.import_module(f"variant{i}")
    out = []
    for load, c in bench.SCORED:
        if load.startswith("verify") and c["layer"] == "full" and load != "verify 1":
            out.append(f"{load} {bench.run_shape(load, c):7.1f}")
    print(f"{name:35s} " + "  ".join(out), flush=True)

"""Timing-only ablations of the Triton split path (results wrong on purpose): writes /tmp/s18/abl_<name>/submission.py.
Time with: LOADS="decode 1,verify 1" python tools/s12/runabl.py /tmp/s18/abl_<name> '[{}]'"""
import os
s = open(os.path.join(os.path.dirname(__file__), "..", "..", "submission.py")).read()
R = {
    "nomerge": [("    if fuse:\n        return out\n    hb =", "    return out\n    hb =")],
    "notiles": [("    hi = tl.minimum(lo + chunk, seq)\n    pt_row", "    hi = lo\n    pt_row")],
    "nonew": [("    if split == 0:\n        # The block's own tokens", "    if split == -1:\n        # The block's own tokens")],
}
R["notiles_nomerge"] = R["notiles"] + R["nomerge"]
R["notiles_nomerge_nonew"] = R["notiles"] + R["nomerge"] + R["nonew"]
for name, reps in R.items():
    t = s
    for a, b in reps:
        assert t.count(a) == 1, (name, a)
        t = t.replace(a, b)
    os.makedirs(f"/tmp/s18/abl_{name}", exist_ok=True)
    open(f"/tmp/s18/abl_{name}/submission.py", "w").write(t)

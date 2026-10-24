"""Timing-only ablations of the Triton split path (wrong results on purpose): /tmp/s38/<name>.py.
Time with tools/s36/abc.py /tmp/s38/nm.py /tmp/s38/<name>.py rounds 'verify 1 full'."""
import os
s = open(os.path.join(os.path.dirname(__file__), "..", "..", "submission.py")).read()
NM = [("    splits += sep\n    hb = plan.get", "    return out\n    splits += sep\n    hb = plan.get")]
GATHER = [("    s = tl.dot(q, tl.trans(k if QK8 else k.to(DOT)))\n    if MASKED:",
           "    acc += (tl.sum(k.to(tl.float32), 0) + tl.sum(v.to(tl.float32), 0))[None, :]\n"
           "    return m_i, l_i, acc\n    s = tl.dot(q, tl.trans(k if QK8 else k.to(DOT)))\n    if MASKED:")]
NOGATHER = [("    row = (page.to(tl.int64) * PAGE + t_ld % PAGE) * ROW_W + hp0 * D\n    k = tl.load(kc_ptr",
             "    row = ((page.to(tl.int64) * PAGE + t_ld % PAGE) % 64) * ROW_W + hp0 * D\n    k = tl.load(kc_ptr")]
R = {"base": [], "nm": NM, "gather": NM + GATHER, "nogather": NM + NOGATHER}
for name, reps in R.items():
    t = s
    for a, b in reps:
        assert t.count(a) == 1, (name, a)
        t = t.replace(a, b)
    open(f"/tmp/s38/{name}.py", "w").write(t)

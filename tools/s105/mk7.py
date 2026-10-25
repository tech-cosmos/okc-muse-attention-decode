"""s105 LZ 7 = peeled exact first tile, then a fixed reference max (no alpha, no acc rescale, no vote: c2's loop), and
once per segment: rows whose acc went non-finite (FP16 P overflow: needs > ~16 binades of growth past the first tile's
max; <= 6.5 on every test and scored input, tools/s105/growth.py) store m = +inf -> _combine_bal REDO recomputes them
exactly. Built from /tmp/s105/c2.py (s104 mk.py) into /tmp/s105/lz7.py."""
import re
D = "/tmp/s105/"
s = open(D + "c2.py").read()
s = s.replace("            m_i = gl.full([M], 0.0, gl.float32, layout=gl.SliceLayout(1, S_L))\n            acc = gl.zeros([M, D], gl.float32, layout=O_L)\n            l8",
              "            m_i = gl.full([M], -1.0e30, gl.float32, layout=gl.SliceLayout(1, S_L))\n            acc = gl.zeros([M, D], gl.float32, layout=O_L)\n            l8", 1)
a = s.index("            for i in range(ntiles):\n                s = (it + i) % NS\n                mbarrier.wait(ready")
b = s.index("                mbarrier.arrive(empty.index(s))\n", a) + len("                mbarrier.arrive(empty.index(s))\n")
loop = s[a:b]
body = loop[len("            for i in range(ntiles):\n"):]
FIX = "                    p = gl.exp2(gl.fma(sc, scale, -m_i[:, None]))\n                    m_new = m_i\n"
assert FIX in body
EX = ("                    m_new = gl.maximum(m_i, gl.max(sc, 1) * scale)\n"
      "                    p = gl.exp2(gl.fma(sc, scale, -m_new[:, None]))\n")
b0 = re.sub(r"\bi\b", "0", body.replace(FIX, EX))
b0 = "\n".join(("    " + l if l else l) for l in b0.split("\n"))
CHK = """            # Rows whose acc went non-finite (FP16 P overflow) store m = +inf: _combine_bal REDO recomputes them.
            asum = gl.convert_layout(gl.sum(acc, 1), gl.SliceLayout(1, S_L))
            m_i = gl.where(asum - asum == 0.0, m_i, float("inf"))
"""
new = "            if ntiles > 0:\n" + b0 + "            for i in range(1, ntiles):\n" + body + CHK
s = s[:a] + new + s[b:]
open(D + "lz7.py", "w").write(s)

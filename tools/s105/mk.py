"""s105: peeled LZ 4 and vote/fallback ablations from /tmp/s105/lz4.py (= exp/lz4 submission.py). Timing with
tools/s54/abl2.py (H100 paired, verify 32 full, real + all-L2). pl = correct; vo*/nv* are timing-only."""
import re
D = "/tmp/s105/"
s4 = open(D + "lz4.py").read()
a = s4.index("            for i in range(ntiles):\n                s = (it + i) % NS\n                mbarrier.wait(ready")
b = s4.index("                mbarrier.arrive(empty.index(s))\n", a) + len("                mbarrier.arrive(empty.index(s))\n")
loop = s4[a:b]
body = loop[len("            for i in range(ntiles):\n"):]
IF0 = """                    if i == 0:
                        p, me, ls = gl.inline_asm_elementwise(LZ_ASM4P, LZ_CON4, [sc, mb_, sb_],
                                                              dtype=(gl.float32, gl.float32, gl.float32),
                                                              is_pure=True, pack=32)
                    else:
                        p, me, ls = gl.inline_asm_elementwise(LZ_ASM4, LZ_CON4, [sc, mb_, sb_],
                                                              dtype=(gl.float32, gl.float32, gl.float32),
                                                              is_pure=True, pack=32)
"""
assert IF0 in body
def one(asm):
    return f"""                    p, me, ls = gl.inline_asm_elementwise({asm}, LZ_CON4, [sc, mb_, sb_],
                                                          dtype=(gl.float32, gl.float32, gl.float32),
                                                          is_pure=True, pack=32)
"""
def peeled(src, asm="LZ_ASM4"):
    b0 = body.replace(IF0, one("LZ_ASM4P"))
    b0 = re.sub(r"\bi\b", "0", b0)
    b0 = "\n".join(("    " + l if l else l) for l in b0.split("\n"))
    new = ("            if ntiles > 0:\n" + b0 +
           "            for i in range(1, ntiles):\n" + body.replace(IF0, one(asm)))
    return src.replace(loop, new)
def nopeel(src, asm):
    return src.replace(loop, loop.replace(IF0, one(asm)))
open(D + "pl.py", "w").write(peeled(s4))
# asm variants: vote only (no fallback code), no vote at all
k = s4.index("    LZ_ASM4 = gl.constexpr(_lz_asm4(False))")
def addasm(src, name, expr):
    return src[:k] + f"    {name} = gl.constexpr({expr})\n" + src[k:]
VO = ("_lz_asm4(False).replace(_lz_asm4(False)[_lz_asm4(False).index('vote.sync'):_lz_asm4(False).index('LZ4_DONE:')], "
      "'vote.sync.all.pred all, ok, 0xffffffff;\\n@all bra LZ4_DONE;\\nmov.f32 ma, 0f7F800000;\\nmov.f32 mb, 0f7F800000;\\n')")
NV = ("_lz_asm4(False).replace(_lz_asm4(False)[_lz_asm4(False).index('setp.le.f32 ok'):_lz_asm4(False).index('LZ4_DONE:')], '')")
k2 = s4.index("    LZ_ASM4 = gl.constexpr(_lz_asm4(False))")
sv = addasm(addasm(s4, "LZ_ASM4V", VO), "LZ_ASM4N", NV)
k = sv.index("    LZ_ASM4 = gl.constexpr(_lz_asm4(False))")
open(D + "pl_vo.py", "w").write(peeled(sv, "LZ_ASM4V"))
open(D + "pl_nv.py", "w").write(peeled(sv, "LZ_ASM4N"))

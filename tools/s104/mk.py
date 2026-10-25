"""s104: build the verify-full lazy-max decomposition variants from exp/lz4's submission.py into /tmp/s104
(timing only; most are INCORRECT on purpose). Run: git show exp/lz4:submission.py > /tmp/s104/lz4.py; python mk.py;
then tools/s54/abl2.py A.py B.py 5 32,8 (H100 paired, real + all-L2)."""
D = "/tmp/s104/"
s4 = open(D + "lz4.py").read()
open(D + "lz3.py", "w").write(s4.replace("\nLZ = 4", "\nLZ = 3"))
PEEL = """                    if i == 0:
                        p, me, ls = gl.inline_asm_elementwise(LZ_ASM4P, LZ_CON4, [sc, mb_, sb_],
                                                              dtype=(gl.float32, gl.float32, gl.float32),
                                                              is_pure=True, pack=32)
                    else:
                        p, me, ls"""
NOPEEL = """                    if True:
                        p, me, ls"""
NOREDO = lambda s: s.replace('REDO=plan.get("lz", LZ) == 4,', 'REDO=False,')
ACC = """                if LZ >= 3:
                    acc = gl.inline_asm_elementwise("""
def no_acc(s, after):
    j = s.index(ACC, s.index(after))
    return s[:j] + """                if LZ == 4:
                    acc = acc
                elif LZ >= 3:
                    acc = gl.inline_asm_elementwise(""" + s[j + len(ACC):]
# d1: LZ 4 without the acc *= alpha asm
open(D + "d1.py", "w").write(no_acc(s4, "l4 = l4 * alpha + gl.reduce(ls, 1, _gw_pick)"))
# d3 / lz4nr: no in-loop i == 0 branch (first tile overflows -> m = inf; REDO off so the merge doesn't recompute)
open(D + "lz4nr.py", "w").write(NOREDO(s4))
open(D + "d3.py", "w").write(NOREDO(s4.replace(PEEL, NOPEEL)))
# c2: no-max ceiling on the same base: P = exp2(s * scale - m) with m fixed at 0, l8 sums, no alpha, no acc rescale
M0 = "            m_i = gl.full([M], -1.0e30, gl.float32, layout=gl.SliceLayout(1, S_L))\n            acc = gl.zeros([M, D], gl.float32, layout=O_L)\n            l8 = gl.sum(gl.reshape(gl.zeros([M, BN], gl.float32, S_L), [M, BN // 8, 8]), 1)\n            l4 ="
s = s4.replace(PEEL, NOPEEL).replace(M0, M0.replace("-1.0e30", "0.0"))
s = s.replace("""                    if True:
                        p, me, ls = gl.inline_asm_elementwise(LZ_ASM4, LZ_CON4, [sc, mb_, sb_],
                                                              dtype=(gl.float32, gl.float32, gl.float32),
                                                              is_pure=True, pack=32)
                    m_new = gl.reduce(me, 1, _gw_pick)""", """                    p = gl.exp2(gl.fma(sc, scale, -m_i[:, None]))
                    m_new = m_i""")
s = s.replace("                    l4 = l4 * alpha + gl.reduce(ls, 1, _gw_pick)",
              "                    l8 = l8 + gl.sum(gl.reshape(p, [M, BN // 8, 8]), 1)")
s = no_acc(s, "l8 = l8 + gl.sum(gl.reshape(p, [M, BN // 8, 8]), 1)")
s = s.replace("            if LZ == 4:\n                # Quad-sum", "            if False:\n                # Quad-sum")
open(D + "c2.py", "w").write(s)
# c2bf / c2bfa: c2 with BF16 P and V (V via Triton's fp8->bf16, or a 9-instruction PRMT/LOP3 asm, unscaled)
L = s.split("\n")
idx = [n for n, l in enumerate(L) if "vh0.index(sj).store(vt.slice" in l or "vh1.index(sj).store(vt.slice" in l][-2:]
idx += [n for n, l in enumerate(L) if "p16 = gl.convert_layout(p.to(gl.float16), P_L)" in l][-1:]
idx += [n for n, l in enumerate(L) if "= gl.allocate_shared_memory(gl.float16, [NS, BN, D], SH16)" in l][-2:]
for n in idx:
    L[n] = L[n].replace("gl.float16", "gl.bfloat16")
bf = "\n".join(L)
open(D + "c2bf.py", "w").write(bf)
asm = ('{ .reg .b32 m, t0, t1, s0, s1; and.b32 m, $2, 0x7f7f7f7f; prmt.b32 t0, m, 0, 0x1404; prmt.b32 t1, m, 0, 0x3424; '
       'shr.u32 t0, t0, 4; shr.u32 t1, t1, 4; prmt.b32 s0, $2, 0, 0x1404; prmt.b32 s1, $2, 0, 0x3424; '
       'lop3.b32 $0, t0, s0, 0x80008000, 0xf8; lop3.b32 $1, t1, s1, 0x80008000, 0xf8; }')
for v in ("vh0", "vh1"):
    old = f"{v}.index(sj).store(vt.slice({'0' if v == 'vh0' else 'D'}, D, 1).load(VL).to(gl.bfloat16))"
    src = old[len(f"{v}.index(sj).store("):-len(".to(gl.bfloat16))")]
    bf = bf.replace(old, f'{v}.index(sj).store(gl.inline_asm_elementwise("{asm}", "=r,=r,r", '
                         f'[{src}.to(gl.int8, bitcast=True)], dtype=gl.bfloat16, is_pure=True, pack=4))')
open(D + "c2bfa.py", "w").write(bf)

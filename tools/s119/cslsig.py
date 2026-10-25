"""s119: s91's csl 'sig'/'tree' owner-merge variants re-run on the current per-KV-head csl layout
(owner branch is now `warp == 0 && rank < 8`). VAR=sig[,tree] python tools/s119/cslsig.py [rounds]"""
import os, sys
sys.path.insert(0, "/home/daytona/repo/tools/s91"); sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import cslvar
orig = cslvar.patch
def patch(src, var):
    i = src.index("namespace vtk {")
    return _patch(src[:i], var) + src[i:]
def _patch(src, var):
    src = src.replace("    if (warp == 0 && rank < 8) {\n        asm volatile(\"{\\n\\t.reg .pred p;",
                      "    if (warp == 0) {\n        asm volatile(\"{\\n\\t.reg .pred p;")
    src = orig(src, var)
    src = src.replace("    if (warp == 0) {\n", "    if (warp == 0 && rank < 8) {\n")
    return src
cslvar.patch = patch
g = dict(vars(cslvar)); g["__name__"] = "__main__"; g["patch"] = patch
exec(compile(open(cslvar.__file__).read().split('if __name__ == "__main__":')[1].replace("\n    ", "\n"), "cslmain", "exec"), g)

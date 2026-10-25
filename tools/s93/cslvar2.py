"""s93: more csl source variants (on top of s91's cslvar harness). VAR=ptsh python tools/s93/cslvar2.py [rounds]"""
import os, sys
sys.path.insert(0, "/home/daytona/repo/tools/s91"); sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import cslvar

OLD_SLOTS = r"""    _Pragma("unroll") for (int j = 0; j < 4; ++j) {                                                            \
        int t = min(max(base + w0 + 8 * j + g, 0), tmax);                                                      \
        kslot[j] = PAGE == 1 ? (long)ptr[t] : (long)ptr[t / PAGE] * PAGE + t % PAGE;                           \
    }                                                                                                          \
    _Pragma("unroll") for (int i = 0; i < 8; ++i) {                                                            \
        int k = 8 * (i >> 1) + 2 * c + (i & 1);                                                                \
        int t = min(max(base + w0 + k, 0), tmax);                                                              \
        vslot[i] = PAGE == 1 ? (long)ptr[t] : (long)ptr[t / PAGE] * PAGE + t % PAGE;                           \
    }                                                                                                          \
"""
NEW_SLOTS = r"""    {                                                                                                          \
        int t = min(max(base + w0 + lane, 0), tmax);                                                           \
        int e = PAGE == 1 ? ptr[t] : ptr[t / PAGE] * PAGE + t % PAGE;                                          \
        _Pragma("unroll") for (int j = 0; j < 4; ++j) kslot[j] = __shfl_sync(0xffffffff, e, 8 * j + g);        \
        _Pragma("unroll") for (int i = 0; i < 8; ++i)                                                          \
            vslot[i] = __shfl_sync(0xffffffff, e, 8 * (i >> 1) + 2 * c + (i & 1));                             \
    }                                                                                                          \
"""
orig_patch = cslvar.patch
def patch(src, var):
    if "ptsh" in var:
        assert src.count(OLD_SLOTS) == 1
        src = src.replace(OLD_SLOTS, NEW_SLOTS)
    if "nc" in var:  # gathers through the non-coherent path with no L1 allocation
        a = "kr[j] = *(const uint4*)(kc + kslot[j] * (HKV * D) + h * D + 16 * c);"
        b = "kr[j] = ldg16(kc + kslot[j] * (HKV * D) + h * D + 16 * c);"
        a2 = "vr[i] = *(const uint2*)(vc + vslot[i] * (HKV * D) + h * D + 8 * g);"
        b2 = "vr[i] = ldg8(vc + vslot[i] * (HKV * D) + h * D + 8 * g);"
        assert src.count(a) == 1 and src.count(a2) == 1
        src = src.replace(a, b).replace(a2, b2)
        hdr = "__device__ __forceinline__ float bf(unsigned short x)"
        src = src.replace(hdr, r'''__device__ __forceinline__ uint4 ldg16(const unsigned char* p) {
    uint4 r; asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p)); return r; }
__device__ __forceinline__ uint2 ldg8(const unsigned char* p) {
    uint2 r; asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p)); return r; }
''' + hdr, 1)
    return orig_patch(src, var)
cslvar.patch = patch
if __name__ == "__main__":
    import runpy
    sys.argv = [cslvar.__file__] + sys.argv[1:]
    g = dict(vars(cslvar)); g["__name__"] = "__main__"; g["patch"] = patch
    exec(compile(open(cslvar.__file__).read().split('if __name__ == "__main__":')[1].replace("\n    ", "\n"), "cslmain", "exec"), g)

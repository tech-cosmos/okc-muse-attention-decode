# H200 variant harness (session 77)

The judge's check output includes a per-kernel GPU-time profile for every scored shape (profiles/*.txt). Branch
**exp/h200diag** uses that to time many variants of the verify-full kernel on the H200 in ONE check:

- `_DIAG_GW` in submission.py: (kernel name, Gluon constexpr overrides, flush first). Each entry launches a renamed
  copy of `_gw_verify` (`_renamed()`: same code, new kernel name) on scratch partials after the real computation,
  optionally after an L2 flush (2 GB torch.sum). ABL bits for ablations: 1 no gathers, 2 no V convert, 4 consumers
  idle, 8 no softmax.
- `_DIAG_NV`: (namespace tag, macro overrides) of the session-71 native kernel (tools/s77/nverify_src.cu, macros
  NV_ABL/NS/PREGS/CREGS/LAZY/LAG/PFD/TR), one load_inline at import, launched WITHOUT PDL (with PDL the copies
  overlap the previous kernel and their times are garbage: check c3c16ec6).
- Keep a baseline copy at the start and at the end: copies of the same kernel agree to ~1-2 us within one run.
- The check's score is meaningless (extra work in the timed call); never submit this branch. Log it as a diagnostic.
- To time variants of other kernels, copy the pattern (`_renamed(kernel, name, triton.jit or gluon.jit)`).

H200 facts measured with it (session 77, verify 32 / 16 full, us): first real _gw_verify 210-214 / 110-114;
flushed copies 208-210 / 108-110; warm copies (no flush) 202-204 / 104; producer-only (consumers idle, warm) 171-177
/ 87-89; gathers only 163-171 / 85-88; consumer-only (no gathers, no convert) 170-173 / 88-91; all-L2 (page_table %
4096) 190-193 / 98-100; no softmax 175-180 / 92-94; no V convert 195-199 / 101-103. Page-table L2 prefetch (ptpf):
no gain. TLB touch (1 B per 64 KB of K/V before the kernel): -1 us (noise).

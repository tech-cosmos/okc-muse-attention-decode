# NOTES-agent2 (second Claude agent: batch 16/32 sliding layers; newest at the bottom)

Brief: AGENT2.md. Shapes: decode 16/32 sliding, verify 16/32 sliding. Tools in tools/a2 (all use ~/repo2 paths):
- `tools/a2/abp.py "<load> <layer>" '<plan A json>' '<plan B json>' [rounds]`: paired H100 A/B of plan overrides
  (alternating order, median of B-A). Use this, not single sweeps (sweep noise ~0.1-0.3 us).
- `tools/a2/chk.py`: correctness on 12 extra configs that actually hit the batched sliding plans (the judge's 16 tests
  only reach verify sliding at batch <= 5, i.e. the `heads <= 32` plan branch, never the batch-32 one).
- `tools/a2/sass.py '<load>' <layer> /tmp/prefix [kernel]`: dumps ttgir/ptx/cubin/SASS + regs of the Triton kernel.
- `tools/a2/one.py '<load>' <layer>`: runs a shape 3x (for `flock /tmp/okc-gpu.lock ncu ... python tools/a2/one.py`).

## Session 1 (2026-10-25 04:09 - ~05:20 UTC), branch agent2-exp/maskif

Baseline (H100 bench, main e8a1f73 code): decode 16/32 sliding 9.60/15.17, verify 16/32 sliding 14.27/18.96 us.
H200 (scored 8aec5993): decode 16 8.19 (csl, one kernel 7.0), decode 32 13.97 (_attn_split 10.8 + _combine_hb 1.5),
verify 16 12.75 (9.2 + 2.1), verify 32 17.72 (13.8 + 2.5). Per-us value (load share / 6): decode 16 0.42 %/us,
verify 16 0.31, decode 32 0.23, verify 32 0.18.

Plan facts worth knowing (they are easy to misread in _plan): verify 16 sliding has heads = 32, so it takes the
`heads <= 32` branch: splits 8, block_n 64, 4 warps, stages 5, no peel (256 programs, 4 tiles each). Verify 32
sliding: the `heads >= 64` branch, splits 4, block_n 64, stages 5 (256 programs, 8 tiles each). Decode 32 sliding:
splits 8, block_n 32, 2 warps, stages 5, peel (512 programs, 8 tiles of 32). Decode 16: native csl.

### ncu / SASS of verify 32 sliding `_attn_split` (H100, ncu clocks 1.45 GHz)

- 121 regs, 33 KB smem, grid 256 x 128 threads: 1.9 CTAs/SM = 7.6 warps/SM (~2 per scheduler). Issue: 0.38 inst
  per scheduler-cycle; stall cycles per warp: long_scoreboard 5.5K, selected 4.4K, wait 4.1K, barrier 2.8K,
  short_scoreboard 1.3K, not_selected 1.3K. 4.52M warp-instructions = 4416 per warp.
- Main loop = 445 SASS instructions per 64-key tile per warp: QGMMA 64x64x32 x2 (QK in FP8), ~130 instructions of
  per-element masks (64 ISETP + 64 FSEL: BOTH the window mask and the `t < hi` mask on EVERY tile, since verify plans
  without peel run every tile through the MASKED path, and even the peeled path applies the window mask to all tiles
  because WIN = WINDOW for L > 1), softmax, V FP8->FP16 convert through smem, HGMMA 64x64x16 x4 (P FP16 from regs),
  4 BAR.SYNC, LDGSTS gathers.

### Win: MIF (masks only on the tiles that need them) + PREQ (agent2-exp/maskif 6c6a87a)

- MIF: one loop, the score masks inside a runtime scalar `if need:` (need = tile reaches past hi, or tile start <=
  seq + L - 1 - WINDOW, i.e. split 0's first tile). Loop body 445 -> ~316 instructions. H100 paired (abp.py, 11
  rounds): **verify 32 sliding -0.61 us (11/11)**, verify 16 sliding -0.24 (10/11). (Notes s14 said "handling that
  tile apart lost"; that was a peeled copy of the tile, which breaks the pipeline; a runtime branch around the masks
  only does not.)
- PREQ 1: the L2 prefetch of the first tile's K/V rows placed AFTER the query/cos/sin loads are issued and BEFORE
  their norm/RoPE math (split `_norm_rope` into `_norm_rope_ld` / `_norm_rope_fin`). verify 16 another -0.13
  (9/11); verify 32 tie (+0.11 with preq 2, -0.02 with 1); decode 32 tie/loss (preq 1/2/4 +0.1/+0.2/+0.7).
- Plans: verify 16 (heads <= 32 branch, non-tiny, L > 1): mif + preq 1; verify 32 (heads >= 64 verify): mif.
  Batch 3-5 verify sliding tests now also run mif + preq (tests 16/16, chk.py 12/12, errors unchanged).

### H200 check a27e9a524eb94585 (6c6a87a): 2.376x [2.341, 2.412], 16/16

verify 32 sliding **16.64 us** (scored 8aec5993: 17.72; 7f45c5c6: 17.42), verify 16 sliding **12.49** (12.75 /
12.91). Untouched shapes moved by noise (verify 1 sliding 6.50 vs 6.91, decode 32 sliding 13.72 vs 13.97), so the
overall +0.9% over 2.354 is partly noise; verify 32's -1.1 us is outside per-shape noise. Listed in COORD "Ready to
merge" (agent 1 merges + submits). Budget: agent2 1 check today (account 10/20 checks, 3/10 scored after it).

### Dead (H100, this session)

- verify 32/16 sliding: fuse (in-kernel last-program merge) +2.9/+5.4 us; dual (2 KV-head chains per program) with
  splits 4/8 and block 32/64: +4..+6; hpp 2 (both heads per program, block-diagonal) with warps 4/8, splits 4-16:
  +4..+22; stages 4 (+3!) / 6 (tie); splits 8: +1..+3. decode 32 sliding: fuse +3.1; block_n 16/64, warps 1/4,
  splits 4/16, hpp 2: all within +-0.3 (plateau).

### Where the remaining time is (for the next session)

- Decode 32 sliding Triton loop: 315 SASS instructions per 32-key tile per warp (2 warps, mma.sync HMMA.16816, ~10
  BAR.SYNC per tile), prologue ~1230 instructions; ~4M warp-instructions = ~4.4 us of pure issue per SM at 100%
  (kernel 10.8 us H200). csl's one-shot structure is ~3x leaner per key but at batch 32 its register-held gathers
  (80 regs x 256 thr x 512 CTAs) don't fit one wave, and 16-CTA clusters cap at 28 resident (s75).
- Verify sliding: consumer-chain bound (s75 all-L2 attn 9.1/13.4 us at verify 16/32). Triton runs 1 WG per CTA,
  2 CTAs/SM. A native design with 2 consumer WGs per CTA (one per KV head, so gathers are full 128-B rows, the s71
  lesson) and 2 CTAs/SM would double warps/SM at the same partial count.

### Next ideas (ranked)

1. (done) MIF/preq confirmed on the H200 and listed under COORD "Ready to merge".
2. Native verify-sliding kernel `nvs` (B 16/32): CTA = (request, key chunk) with 2 consumer WGs (KV head 0/1) and no
   producer WG: the WGs issue their own cp.async full-row gathers (8 lanes x 16 B per 128-B row, page-table entries
   for the next tiles staged in smem), QK^T as SS wgmma m64n64k32 E4M3, V FP8->FP16 into smem + HGMMA with P FP16 in
   registers (what Triton does, precision >= SGLang), window mask only on the first tile (as MIF), new tokens in the
   last chunk's CTA as an extra 8-key tile, partials to global + existing `_combine_hb` (PDL) first; cluster DSMEM
   merge later. Load order as csl: seq/q/gate first, PT, K/V, then q norm/RoPE. Bases: tools/s71/nverify.cu (wgmma
   descriptors, cp.async SW128 staging, FP8 V^T), tools/s72/csl.cu (cluster merge, load order). Target tile rate
   ~3 tiles/us/SM (gw/nverify level) vs Triton ~1.8: ceiling -3.4 us verify 32, -1.7 verify 16 (H100).
3. Decode 32: smem-staged (cp.async full-line, swizzled) csl-style CTAs in 8-CTA clusters (256 keys/CTA, 64 KB smem,
   2 CTAs/SM = one wave), compute per 32-key chunk as each cp.async group lands. Risky: s73 csl8 (similar) lost
   16.8 vs 15.9 H100 (compute 2.7 + merge 2 us exposed after the last load).

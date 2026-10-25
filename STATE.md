# STATE (short current picture; NOTES.md is the long log: grep it)

Updated 2026-10-25 ~05:30 UTC (session 90 = unattended session 23 of the #1-push brief).

**s90: merged agent2 maskif (mif + preq, verify 16/32 sliding) -> main ad2116c; SCORED 0339a5f6 2.364x = NEW ACCOUNT
BEST.** Verify 1 full probes: compute-latency-bound in Triton (warm 14.2 vs cold 15.45 us; 6.2 us fixed + 0.111 us/1k
keys; doubling warps/SM raises CPI 4.5 -> 7.1, IPC +15%); FP8 P@V in Triton dead (+10 us). NOTES s90.

**s89: native wgmma verify-1-sliding single kernel (tools/s89/vsw.cu): correct, dead, no checks.** H100 8.0-8.2 vs
main 7.6 us. Compute is fine (wgmma tile ~1.0 us after the gathers land); the wall is the cluster merge: each CTA
must send a 16 KB FP32 64-row partial over DSMEM (~1.3 us, ~12 B/ns/SM, matches the s74 microbench), and an L2
exchange is far worse (13.2). Third native verify-1-sliding kernel to lose (s72, s84, s89). NOTES s89.

**s88: batch-1 full probes, all dead, no checks** (tools/s88). H100: decode 1 full merge exposure 2.9 us = acc loads 1.0
+ fixed 1.9; native gather floor for its bytes 9.5 us vs `_gd_full` attention 11.7. Dead: `_gd_full` hoisted prologue
(deeper rings still lose), `_combine` num_ctas 2/4 (+3.4), verify 1 full dual/splits 191-255/peel, uniform-branch tail
mask (-16% loop instructions, tie). Since 04:09Z agent2 (~/repo2) owns batch 16/32 sliding and gets >= 8 checks/day;
merge its COORD "Ready to merge" wins. NOTES s88.

**s87:** plan knobs, `_gd_full` ptpf, gbal K/V L2 prefetch-ahead: all dead on H100 (NOTES s87).

**s86: SCORED 8aec5993 2.354x (then best) (main e8a1f73, code 07a06b3).** Bisected s85's verify 1 sliding
SPEC loss: SPEC alone (no pre 2) also lost on the H200 (diag 6e8ac91c: 9.24 us); the cause is the unexecuted
out-of-line `noinline` redo copy (2x PTX). Loop form (one body in `while go:`, seq loop-carried) + pre 2: H100 -0.29,
H200 check e1620b92 2.359x (verify 1 sliding 7.32), scored 8aec5993 2.354x (verify 1 sliding 6.91). `_gd_full` loop
form loses +1.2 H100 (keeps noinline, which is fine in Gluon); verify 1 full SPEC +0.35 H100 (off). NOTES s86.

**s76-s85 (NOTES has the detail):** s85 SPEC (batch-1 seq from table width, on main); s84 native vsl lost, bucketing
gate failed; s80 H200 decode 32 full bytes dense 144.7 / random ~164 us; s79 LZ 3 lazy max (on main); s77 diag harness.

## Board and best run

- Board 2026-10-25 05:25Z: #1 pjmullin3 2.910x, #2 colinlikescode 2.844x, #3 2.643x; **ours 2.364x (#22)** (scored
  **0339a5f6**, main ad2116c; previous 8aec5993 2.354x). Need ~+23% for #1. Table below is the older 7f45c5c6 run (NOTES s90 has 0339a5f6's).
- Per-shape H200 (7f45c5c6, us, ours / SGLang / speedup):

| load | sliding (x46) | full (x16) | load speedup |
|---|---|---|---|
| decode 1 | 6.05 / 21.09 / 3.48x | 13.70 / 31.46 / 2.30x | 2.96 |
| decode 16 | 10.72 / 27.33 / 2.55x | 86.52 / 202.26 / 2.34x | 2.39 |
| decode 32 | 13.86 / 31.58 / 2.28x | 161.70 / 386.54 / 2.39x | 2.37 |
| verify 1 | 7.17 / 23.06 / 3.22x | 16.50 / 36.22 / 2.20x | 2.76 |
| verify 16 | 12.91 / 29.62 / 2.29x | 115.81 / 206.83 / 1.79x | 1.91 |
| verify 32 | 17.42 / 34.85 / 2.00x | 213.70 / 392.37 / 1.84x | 1.87 |

- H200 profile (7f45c5c6): verify 16/32 full `_gw_verify` 109.2 / 209.8 us + `_combine_bal` tail; decode 32 full
  `_gd_full` 162.5 (4.0 TB/s, near the gather floor). Verify full is consumer-compute-bound on the H200.

## Code state

- main = ad2116c (= agent2 6c6a87a code = scored 0339a5f6 2.364x).
- Dispatch: decode sliding B<=16 -> native `_CSL` (one 16-CTA cluster per request, DSMEM st.async merge); verify full B*Hkv>4 -> Gluon `_gw_verify` gbal (warp-specialized: 1 producer WG gathers + converts V
  to FP16, 2 consumer WGs = one per KV head, 2 CTAs/SM); decode full -> Gluon `_gd_full`; sliding + small -> Triton
  `_attn_split` + PDL merges.

## Budget (UTC day 2026-10-24; API: `okc problems --json`)

- 2026-10-25: checks 10/20 at s89 start (s89 used none; agent2 gets >= 8 of the rest); earlier: checks 9/20 (s86: 6e8ac91c diag 2.302x, e1620b92 2.359x; scored 8aec5993 2.354x = 3/10 scored) (s85: f51b8c4b 2.300x, 297d496a 2.347x; c5888ee5 diag, 577c5b9d 2.353x, 50c3fc4d + 025c8861 bandwidth diags, f4199324 LZ 4 2.349x tie), scored 2/10 (8774bdfc 2.312x, 9f8e594b 2.350x). Older 2026-10-24 notes below.
  2.328x, decode 32 sliding 13.27 vs 13.42 us = tie, not merged). Resets at
  00:00 UTC: 20 checks / 10 scored on 2026-10-25. Claude may use all account checks (human, a084e9c).

## Current direction

See CLAUDE.md "Current direction" (lead agent, e76a294): batch-1 shapes first, verify full only with a structural
idea, physical-order decode full after a cheap multi-split prototype. (s69's native verify-full consumer microbench
numbers: NOTES s69.)

## Ranked ideas

000000. (s80; s84: GATE FAILED, bucketing pass 42+ us H100, see NOTES s84) Physical-order decode full (NOTES s80 design): prototype first the multi-split pass (<= ~5 us?) and
   mixed-request mma.sync compute under a dense 256-row stream. ~+1-1.5% overall. Below the verify-full consumer.

00000. (s79) LZ 3 follow-up: sum-based check (no FMNMX), power-of-two per-warp fallback, inf-only redo launch
   (NOTES s79 Next 1). Then the producer half.

0000. (s78) Exact lazy max in the verify-full consumer at ~no cost: -27/-16 us H200 ceiling (verify 32/16 full) =
   ~+3% overall. Branches exp/lazymax (Gluon SPEC variants), exp/lazydiag (diag harness), exp/nvlazy (native). NOTES s78.

000. (s84: native verify-1-sliding cluster kernel built and lost, NOTES s84) (s75) Verify sliding is consumer-chain-bound, not DRAM-bound (H100 all-L2 attention verify 32/16 sliding 13.4/9.1 vs
   real 16.5/11.2 us; Triton ~1.8 64-key tiles/us/SM vs Gluon gw 2.9 on verify full). gw_verify itself loses at
   sliding size (fixed costs). The open design is a native wgmma verify-sliding kernel with csl-style low fixed
   cost (no warp-spec setup, cluster DSMEM merge), 2-4 pipelined tiles per CTA: ~1.5-2% overall if it reached gw's
   tile rate. NOTES session 75.
00. (s74) H100-tied plan variants lose or tie on the H200 (4 of 4: verify 16 sliding splits 4 loses, csl TLB warm,
   decode 16 full dbal, verify 1 sliding stages 3 tie). Per-shape judge noise ~+-2%: don't spend checks on ties.
   Measured ceilings (NOTES s74): bulk-copy row gathers 2.35 TB/s (worse than LSU 2.77); in-cluster verify-volume
   merge ~0.5-0.7 us DSMEM at batch 1 (native verify-1-sliding upside <= ~1%).
0. (s73) One-shot native kernels (csl8 for decode 32 sliding, cfl for decode 1 full) LOSE on the H100: compute and
   merges land after the last gather (decode 32: loads 12.1 -> 16.8 us; decode 1 full: loads 12.2 -> 19.5). Native
   wins need a pipelined multi-tile loop like Triton/Gluon's, not one shot. NOTES session 73.
0b. (s72) Cluster kernel follow-ups: batch 32 decode sliding (csl2 NC 8 needs register/load-order fixes); shave csl
   (compute ~0.8, merge ~0.8, cluster launch ~0.45 us on H100). Verify sliding on mma.sync clusters LOST (7.6 vs 5.5 us
   H100: 8x partial volume + emulated FP8 mma); would need wgmma + few key splits. NOTES session 72.

1. Native verify-full kernel BUILT (session 71, branch exp/nverify 5bc0e6a, tools/s71/nverify.cu): correct, but H200
   check 19485acf kernel 209.9/112.0 us vs Gluon 209.8/109.2 at verify 32/16 = tie/loss. Producer (gathers + FP8 V
   transpose) and consumer contend for issue; see NOTES session 71 for the measured levers. Resume only with a
   concrete way to cut producer instructions (now 4 LDSM + 16 PRMT + 4 STSM + 8 cp.async + 4 LDS per thread/tile).
2. Sliding shapes: no cheap lever left in Triton (cluster merges dead, s70); a native sliding kernel would have to
   beat `_attn_split` + PDL merge on the H200 (Codex's native sliding didn't).
3. Merge tail on verify full (~7 us on the H200 at verify 16/32 = ~0.8% overall at most).

## Queued for checks

- Nothing queued: main ad2116c = scored 0339a5f6 (2.364x). Submit a new main only if its check beats 2.364.

## Dead-ends index (NOTES.md line refs approximate; grep the session)

- s90: FP8 P@V in Triton _tile (verify 1 full +10 us), verify 1 full block_n 128 / block_n 32 st 12 / warps 8.

- s89: native wgmma verify-1-sliding kernel (vsw: st.async / bulk DSMEM / L2 exchange); cross-CTA merge at verify
  volume = 16 KB/SM of DSMEM ~1.3 us, which kills any single-kernel verify sliding design.

- s88: `_gd_full` HOIST prologue, `_combine` num_ctas 2/4, verify 1 full dual / splits 191-255 / peel / CM tail mask.
- s87: plan knobs verify 1 full / decode 1 full (fewer splits, deeper stages), `_gd_full` ptpf, gbal producer K/V
  L2 prefetch-ahead (pfd, incl. evict_last).
- s86: noinline redo copy in Triton _attn_split (H200 +2 us cold), loop-form SPEC in Gluon _gd_full (+1.2 H100),
  verify 1 full SPEC (+0.35 H100).
- s85: SPEC in _attn_split for verify 1 sliding (H200 judge +2.4 us, unexplained) and verify 1 full (H100 +0.1..0.2).
- s84: native verify-1-sliding cluster kernel (vsl, NPH 8/16), physical-order bucketing pass (bucket.cu, 42+ us),
  verify 1 sliding 64 splits.
- s83: TV (tensor-core V convert in _gw_verify, exp/tv), decode 1 full merge cw/csb/splits.

- s82: F2FP/MUFU pipe sharing (none), verify 1 full on gw_verify (tie/loss), verify full 1 CTA/SM incl. BN 128.

- s81: LZ 4 (exp/lz4; H200 tie), consumer-side redo (spills / noinline unsupported in WS partitions), tree sums.
- s79: LZ 1 (keep S live for the fallback: spills), LZ 6 (no per-warp fallback: rows grow > 16 binades after tile
  0), tree max (+0.8), 3-input max (not on sm_90), Triton _tile lazy max (exp/lztri, +0.1..+0.5 us).

- s77: page-table L2 prefetch in the gbal producer (ptpf, H200 flushed pairs: tie); TLB touch before verify full
  (noise); speculative max inside nverify (C7512/C7511 at 104 regs).

- s75: cp.async smem-staged gathers in csl (+0.2 us K, +0.3 V at batch 1; and 16-CTA clusters cap at 28 resident even
  at 4 CTAs/SM, so batch 32 can't be one wave); Gluon gw_verify at sliding size (proxy 16.2/21.0 vs Triton
  12.8/19.5 at batch 16/32); verify sliding splits 6/8/12/16, stages 3, warps 8, maxnreg 128.
- s74: nverify at verify 1 full (17.8-20.9 vs 14.8 attn), page-table row prefetch in csl (+1.3 at B16), TLB warm,
  bulk-copy gathers (tools/s74/tgather.cu), native cluster merge for decode 1 full (tools/s74/cmg.cu, +0.56/+6.4),
  verify-volume cluster merge microbench (tools/s74/vmerge.cu).
- s73: csl8 (8-CTA clusters, 2 chunks/warp, cp.async chunk B) decode 32 sliding 16.8 vs 15.9 us; csl at batch 32
  with launch_bounds 4 (spills, 18.3); native decode-1 full cfl with a last-arriver cross-cluster merge 19 vs 15.8.
- Cluster DSMEM merges: from Triton dead (s70), but native with st.async + mbarrier complete_tx WORKS (s72). Pitfalls:
  release cluster barriers (~0.5 us), and loads placed after page-table-dependent ones (in-order issue).

- Native CUDA verify full with FP8 P@V (s71, exp/nverify): H200 tie with Gluon. Lessons: page-table entries off the
  per-tile DRAM chain (smem ring), 8 lanes per 128-B row, NS >= LAG + 2.

- Verify-full Gluon consumer: FA3 pipelining (C7514 serialization, s37), BN 32 (s56), conditional rescale (s18
  exp/wcr), FP16 scores / f16x2 ex2 (s12, s49), FP8 P@V ceiling only -10 us all-L2 (s22), 6 WGs/SM (s42, s51),
  K-ready barrier (s47), V ring split (s46), bank-conflict fix (s45).
- Decode full: near gather floor (s39: dense read 232 vs random 236 us H100).
- Batch-1 sliding: fused last-program merge (s36, s43, s50), Gluon cluster merge (s25: Gluon DSMEM
  codegen was slow; the native push-merge in tools/s69/cmerge.cu is a different, faster pattern).
- Thread-block cluster (DSMEM) merges launched from Triton (s70, exp/cluster): batch-1 sliding +1.7 us, batched
  sliding +3..+6 (cluster launch alone +1.3/+2.3), decode 1 full merge +0.7..+3.2. Dead from Triton/Gluon.
- H100-only sliding wins that lost on the H200: batched sliding hpp 2 (scored 56fca423), Gluon sliding (f3e099c1).

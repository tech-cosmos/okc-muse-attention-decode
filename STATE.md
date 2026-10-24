# STATE (short current picture; NOTES.md is the long log: grep it)

Updated 2026-10-24 ~22:30 UTC (session 74 = unattended session 6 of the #1-push brief).

## Board and best run

- Board 2026-10-24 21:30Z: #1 colinlikescode 2.844x, #2 2.726x, #3 2.624x; **ours #15 2.343x**
  (scored **7f45c5c6**, commit 14355d4 = main's kernel). Need ~+21% for #1.
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

- main = c9b3242 (session 72): 14355d4's kernel + the native cluster decode-sliding kernel `_CSL` (tools/s72/csl.cu)
  for decode sliding at batch <= 16. Checked a6081cf6: 2.322x [2.287, 2.357] (noise on untouched shapes); decode 1
  sliding 5.63 us (was 6.05), decode 16 sliding 8.70 (was 10.72). NOT yet scored: re-check/submit next UTC day.
- Dispatch: decode sliding B<=16 -> native `_CSL` (one 16-CTA cluster per request, DSMEM st.async merge); verify full B*Hkv>4 -> Gluon `_gw_verify` gbal (warp-specialized: 1 producer WG gathers + converts V
  to FP16, 2 consumer WGs = one per KV head, 2 CTAs/SM); decode full -> Gluon `_gd_full`; sliding + small -> Triton
  `_attn_split` + PDL merges.

## Budget (UTC day 2026-10-24; API: `okc problems --json`)

- Account: checks 15/20 used, scored 9/10 used after session 74's checks 5ff29fde, 9a22a5ae (5 checks, 1 scored left
  until 00:00 UTC). Claude may use all account checks (human, a084e9c).

## Current direction (session 69): native verify-full consumer (CUDA, inline-PTX wgmma)

Measured on H100, verify 32 full sizes (tools/s69, tools/s54/abl2.py ablations of `_gw_verify`):
- Gluon kernel all-L2 196 us; producer-only 107; consumer-only (no gathers/convert) 178. Consumer is the limit.
- Native consumer-only microbench (tools/s69/cmicro.cu, wgmma QK FP8 + PV FP16, correct to 2e-4):
  unpipelined 163 us, FA3 intra-WG pipelined 164-166 (no gain), 1 CTA/SM pipelined 164.5.
  No softmax 87; no exp (MUFU) 145; no PV 140-155. MUFU pipe 60% busy, issue 55%, 3.5 warps/scheduler.
- So a native kernel's consumer floor is ~160 vs Gluon 178 all-L2: maybe -15..-20 us on H200 verify 32.
- Poly exp2 offload, P@ones row sums: lose natively too; warp-vote lazy rescale ~-3%. Native verify-full upside is
  ~+1.5-2% overall: not the top lever.
- Sliding gather floor with every row in flight (tools/s69/sgather.cu): decode 1/16/32 = 3.2/6.7/10.9 us (H100), same
  as before. Our sliding shapes are 1.3-8.6 us above it.
- Cluster push-merge (tools/s69/cmerge.cu): one kernel, 16-CTA cluster, each CTA stores its partial rows into the
  owner's smem (st.shared::cluster), one cluster barrier, local merge: 2.32 us vs 3.87-4.03 for partials + separate
  PDL merge kernel (-1.6 us at 16 splits; -0.45 at 4). Pulling via map_shared_rank loads costs ~+1 us (latency).

## Ranked ideas

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

- Nothing new. main c9b3242 = 2.320-2.322x in three checks (a6081cf6, 8bd56b13 kernel-equal, 9a22a5ae near-equal);
  submit only after a real change lifts a check above 2.343.

## Dead-ends index (NOTES.md line refs approximate; grep the session)

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

# STATE (short current picture; NOTES.md is the long log: grep it)

Updated 2026-10-25 ~10:20 UTC (session 105 = unattended session 38 of the #1-push brief).

**s105: main d6cb195 = LZ 10 fixed-reference softmax in _gw_verify (4 exact tiles/segment with +14 binades headroom,
then no max/alpha/rescale/vote; non-finite rows -> m=inf -> _combine_bal REDO, exact). H100 paired vs LZ 3: verify 32
full all-L2 -13.5 us (real -0.6), verify 16 -6.8 (real -0.2). H200 UNVERIFIED (c2-class ceiling was -26/-16). Risk:
REDO ~17 ms if a scored row grows > 30 binades (max seen 23.2). NOTES s105.**

**s104: LZ 4 gap = in-loop branches (s105 confirmed). s103: red on batched decode full (H100 -1.58/-1.09 us, on main, H200
unverified). s102: submitted 8725ba7e 2.460x = account best; full layers bimodal across judge nodes. NOTES s102-s104.**

**Earlier (details in NOTES s91-s101): s101 decoupled V ring in gbal = H200 tie/loss; s100 vtk H200 win (verify 1 sliding 6.8 -> 5.6), launch floor ~2 us/kernel;
s99 physical-row order dead; s98/s97 vtk / nverify probes dead; s96 vtk built; s95 red (fixed-reference-max atomic
merge) for decode 1 full (H200 -0.6 us); s91 csl scored 08bf9391 2.445x.**

## Board and best run

- Board 2026-10-25 09:05Z: #1 pjmullin3 2.910x, #2 colinlikescode 2.844x, #3 2.643x; **ours 2.460x** (scored
  **8725ba7e**, main cc93187 = 1e4c80e code; previous 08bf9391 2.445x).
- Per-shape H200 (8725ba7e, us): decode 1 4.55 / 13.71; decode 16 7.70 / 89.55; decode 32 13.83 / 168.92; verify 1
  5.53 / 16.45; verify 16 12.43 / 113.28; verify 32 15.21 / 209.94 (slow-node full layers). Need ~+18% for #1.
- H200 profile (7f45c5c6): verify 16/32 full `_gw_verify` 109.2 / 209.8 us + `_combine_bal` tail; decode 32 full
  `_gd_full` 162.5 (4.0 TB/s, near the gather floor). Verify full is consumer-compute-bound on the H200.

## Code state

- main = d6cb195 (1e4c80e + s103 batched red/dbal for decode 16/32 full + s105 LZ 10 verify full; both unchecked). 1e4c80e = SCORED 8725ba7e 2.460x.
- Dispatch: verify sliding B=1 -> native `vtk_verify` (s96); verify sliding B>=32 -> agent2's native nvs + _combine_hb; decode sliding B<=16 -> native `_CSL` (one 16-CTA cluster per (request, KV head), 4 warps/CTA, DSMEM st.async merge); verify full B*Hkv>4 -> Gluon `_gw_verify` gbal (warp-specialized: 1 producer WG gathers + converts V
  to FP16, 2 consumer WGs = one per KV head, 2 CTAs/SM); decode full -> Gluon `_gd_full` (batch 1: plan red, atomic merge + `_red_fin`); sliding + small -> Triton
  `_attn_split` + PDL merges.

## Budget (UTC day 2026-10-25; API: `okc problems --json`)

- 2026-10-25 after s103: checks 18/20 (2 left = agent2's), scored 7/10 (8725ba7e from the lead agent's reserve). s103: 0/0.

- Today's judge runs are listed in ~/COORD.md "Judge runs" and NOTES s85-s91. Agent2 keeps >= 8 checks/day (it has used 1).

## Current direction

See CLAUDE.md "Current direction" (lead agent, e76a294): batch-1 shapes first, verify full only with a structural
idea, physical-order decode full after a cheap multi-split prototype. (s69's native verify-full consumer microbench
numbers: NOTES s69.)

## Ranked ideas

0000000. (s105 DONE: LZ 10 on main, it was the in-loop branches) Locate LZ 4's 12 us over the no-max ceiling c2 (H100 all-L2; c2-class was -26/-16 us on the H200 at
   verify 32/16 full = ~+3% overall): truly peeled first tile, vote-only asm. tools/s104/mk.py, NOTES s104.

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

- **FIRST after 00:00Z 2026-10-26: check main d6cb195** (s103 red on decode 16/32 full + s105 LZ 10 on verify 16/32 full);
  submit if it beats 1531.2 us. Per shape: decode 16/32 full lose -> BRED = False; verify 16/32 full lose -> LZ = 3.
- **SECOND: check exp/hdiag 63dd0fd (DIAG, never submit): read `<variant>_gd_full` /
  `_attn_split` / `hfloor` durations from the profile; port per-shape plan winners to main, then check main.**
  (main 1e4c80e was scored in s102: 8725ba7e 2.460x; don't resubmit it unchanged.)

- Idea to try (H100 first): csl at batch 32 (CSL_MAX_BATCH 32; agent2's lane, tell it), csl 8 warps x 16 keys per CTA.

## Dead-ends index (NOTES.md line refs approximate; grep the session)
- s105: any in-loop vote/branch in the verify-full softmax (lz9 +1.9 even with no fallbacks taken); zero-headroom fixed
  reference (REDO storm); 8 exact tiles (-3.4 us vs 1).
- s104: BF16 P/V for verify full (producer fp8->bf16 convert +17.6 us all-L2 even as asm), removing the LZ_ACC asm (+2).
- s103: red on verify full (consumer spills at cregs 96-104), decode 1 red NR != 4, pre-summing atomics (atomics cost ~0),
  _combine_bal hb/cw.
- s101: decoupled V ring / more gathers in flight in gbal (H200 diag tie/loss), FA3 ping-pong, dual-head ILP (microbench), decode 1 full splits.
- s99: per-program physical order in _gd_full (pre-pass >= copy floor; aligned-sort ceiling only -3.3%).
- s98: vtk coalesced page table + shfl (tie), cluster size vs launch floor (none).
- s97: nverify at verify 1 full (gather-bound, ~1 us H200 ceiling), vtk2a re-ablated (base 6.88 > vtk 6.45), vtk disjoint
  windows (7.4: shared lines help).
- s96: vtk2a (2 tokens per cluster, 8.1+), vtkp 8-CTA x 8-warp clusters (7.1 vs 6.5), reciprocal divides in decode csl (tie).
- s95: in-grid arrival counter + last-arriver tail (acq_rel arrivals ~1.3 us, tail waits on the L2 atomic queue);
  atomic merge at verify 1 full (tie) / verify 1 sliding (+0.4); not ported to batched full (merge tails 0.2-4.1 us
  H100 vs >= 2x the atomic volume).
- s94: verify 1 full gfuse / one-pass in-grid merge (18.7 vs 17.9), gw non-bal at verify 1 full (whole tie).
- s93: nvs at verify 1 sliding (S 8/16/32), csl ptsh (coalesced page table + shfl), csl nc gathers.
- s92: native pipelined decode-1-full kernel + `_combine` (nfl: register or cp.async rings, NW 4-16, NCTA 64-256).
- s91: csl owner merge fast reciprocal/sigmoid-early/tree reductions, cluster wait after QK (H100 ties); not i-fetch.
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

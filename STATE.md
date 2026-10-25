# STATE (short current picture; NOTES.md is the long log: grep it)

Updated 2026-10-25 ~16:15 UTC (session 125 = unattended session 58). **s125 (0 checks): NT_AFTER on main 410dcb6 = `_new_tokens` launched after `_gw_verify` (gbal): H100 verify 16/32 full -2.43/-1.87 us (11/11); gbal end spread measured (slowest program 6-12 us after the median); dynamic tail built (exp/dtail) and loses as is.** **s124 (0 checks): no extra graph nodes in any shape; 16-CTA cluster CTAs all start within 0.1 us (vtk launch rate not the issue); NOTES s124.** **s123 (0 checks): pre-reset sanity of main ae1161c: tests 16/16, H100 bench 2.053; `_red_fin` counter-spin merge priced dead (NOTES s123).** **s122 (0 checks): tensor vs MUFU pipe-overlap
microbench (tools/s122/tm.cu): wgmma and MUFU/FFMA from different WGs overlap (10-24% mutual slowdown), so the
verify-full consumer gap is chain scheduling, not a HW pipe limit; NOTES s122. s121 (0 checks): verify 1 full on agent2's nvs
(exp/nvsf) DEAD: H100 cold +0.9 us (warm -0.6). s120: exp/hdiag5 @ 840db1a replaces hdiag4 as queued check 2.**

**Earlier (0 checks each): s119 H100 refs for hdiag4's batch-1 variants (all lose); s118 `_red_fin` 16 warps at batch 1
(main ae1161c, H100 -0.3 us); s114 physical-order decode full CLOSED; s115 judge profile is warm; s116-s117 merges <= 0.3%.**

## Board and best run

- **2026-10-25 12:47Z: e64d3450 = main f2b1c52 (agent2 652f126: nds decode 32 sliding 12.20, nvs verify 16 sliding
  12.06) scored 2.542x, 1481.9 us, rank 8 = ACCOUNT BEST.** Board 13:30Z: #1 2.910x, #7 2.569x.
  Per-shape H200 (e64d3450, us): decode 1 4.44 / 13.57; decode 16 7.53 / 87.66; decode 32 11.93 / 168.19; verify 1 5.74 /
  16.18; verify 16 11.90 / 106.19; verify 32 15.42 / 196.60. vs 8725ba7e: **LZ 10 (s105) WINS verify 16/32 full -7.1/-13.3**,
  fix 1 verify 1 full -0.27, red decode 16/32 full -1.9/-0.7 (full layers on the usual slow node). So main's s103/s105/s106
  changes are H200-verified; the queued 'check main 05efdde' item is done.

- H200 profile (7f45c5c6): verify 16/32 full `_gw_verify` 109.2 / 209.8 us + `_combine_bal` tail; decode 32 full
  `_gd_full` 162.5 (4.0 TB/s, near the gather floor). Verify full is consumer-compute-bound on the H200.

## Code state

- main = 410dcb6 = ae1161c + s125 NT_AFTER (verify 16/32 full, H100 -2.4/-1.9, H200 unverified). ae1161c = 5e3d586 + s118 `_red_fin` 16 warps at batch 1 (decode 1 full, H100 -0.3, H200 unverified). 5e3d586 = f2b1c52 + agent2 7ba35f9 NV_HALF (H200-verified per shape in check 2d0bfdc6; tests 16/16; unscored). f2b1c52 (= agent2 652f126 merged, SCORED e64d3450 2.542x) on top of 05efdde (1e4c80e + s103 batched red/dbal for decode 16/32 full + s105 LZ 10 verify full + s106 fix 1 verify 1 full + SPEC clamp; all unchecked). 1e4c80e = SCORED 8725ba7e 2.460x.
- Dispatch: verify sliding B=1 -> native `vtk_verify` (s96); verify sliding B>=32 -> agent2's native nvs + _combine_hb; decode sliding B<=16 -> native `_CSL` (one 16-CTA cluster per (request, KV head), 4 warps/CTA, DSMEM st.async merge); verify full B*Hkv>4 -> Gluon `_gw_verify` gbal (warp-specialized: 1 producer WG gathers + converts V
  to FP16, 2 consumer WGs = one per KV head, 2 CTAs/SM); decode full -> Gluon `_gd_full` (batch 1: plan red, atomic merge + `_red_fin`); sliding + small -> Triton
  `_attn_split` + PDL merges.

## Budget (UTC day 2026-10-25; API: `okc problems --json`)

- Account at s114: checks 20/20 used (0 left), scored 8/10. Judge runs: ~/COORD.md "Judge runs". Agent2 keeps >= 8 checks/day.

## Current direction

See CLAUDE.md "Current direction" (lead agent, e76a294): batch-1 shapes first, verify full only with a structural
idea, physical-order decode full after a cheap multi-split prototype. (s69's native verify-full consumer microbench
numbers: NOTES s69.)

## Ranked ideas

00000000. (s125) dynamic tail for gbal verify full: BUILT on exp/dtail, LOSES +5..+18 us (per-chunk serial DRAM chain). Revive only with next-chunk grab-ahead + query prefetch (NOTES s125). The target (slowest program 6-12 us after the median) is still there.

0000000. (s111: DEAD) native pipelined verify-full kernel (exp/nvf, NOTES s110-s111).
000000. (s114: CLOSED) physical-order decode full: inversion >= 18 us H100 (s84, s114), windowed L2 sweep dead (s114).
000. (s84: native verify-1-sliding cluster kernel built and lost, NOTES s84) (s75) Verify sliding is consumer-chain-bound, not DRAM-bound (H100 all-L2 attention verify 32/16 sliding 13.4/9.1 vs
   real 16.5/11.2 us; Triton ~1.8 64-key tiles/us/SM vs Gluon gw 2.9 on verify full). gw_verify itself loses at
   sliding size (fixed costs). The open design is a native wgmma verify-sliding kernel with csl-style low fixed
   cost (no warp-spec setup, cluster DSMEM merge), 2-4 pipelined tiles per CTA: ~1.5-2% overall if it reached gw's
   tile rate. NOTES session 75.
00. (s74) H100-tied plan variants lose or tie on the H200 (4 of 4: verify 16 sliding splits 4 loses, csl TLB warm,
   decode 16 full dbal, verify 1 sliding stages 3 tie). Per-shape judge noise ~+-2%: don't spend checks on ties.
   Measured ceilings (NOTES s74): bulk-copy row gathers 2.35 TB/s (worse than LSU 2.77); in-cluster verify-volume
   merge ~0.5-0.7 us DSMEM at batch 1 (native verify-1-sliding upside <= ~1%).
0. (s73, s72) one-shot native kernels lose (compute + merges land after the last gather); verify sliding on mma.sync
   clusters lost (8x partial volume). Cluster kernel follow-ups mined out (s91-s109). NOTES s72-s73.
1. Native verify-full kernels (s71 nverify, s111 nvf) tie/lose Gluon `_gw_verify`: resume only with a way to cut
   producer instructions or fit a pipelined consumer at 2 CTAs/SM (NOTES s71, s110, s111).
2. Sliding shapes in Triton: no cheap lever left (s70); batch 16/32 sliding = agent2's lane.
3. Merge tail on verify full: H100 exposure 3.7-3.8 us (s117), in-grid merge net ~2 us = ~0.3% overall: not worth it.

## Queued for checks (exact; sandbox is deleted ~17:45Z 2026-10-25, everything is pushed to origin)

1. **FIRST after 00:00Z 2026-10-26: check main @ 410dcb6 code** (= ae1161c + NT_AFTER; H100 bench 2.057; if verify 16/32 full lose vs e64d3450's 106.19/196.60 set NT_AFTER = False) (later main commits are notes only). Per shape vs
   e64d3450 (scored 2.542x, 1481.9 us): verify 16 sliding changed (agent2 NV_HALF, H200 11.63 vs 11.90 in
   2d0bfdc6); verify 32 sliding uses NV_HALF too (tie 15.43 vs 15.42); decode 1 full `_red_fin` 16 warps (s118, expect
   ~-0.2 vs 13.57; revert RF_NW1 to 4 if it loses > 0.3); all other shapes = e64d3450 code. Submit if the
   check beats 1481.9 us (and tests pass).
2. **exp/hdiag5 @ 840db1a (diag, NEVER submit; = exp/hdiag4 8b557ec + main 705d9d2 + exp/ua `d1ua505`/`d1ua511` decode 1 full
   variants; hdiag4 = hdiag3 + main 5e3d586 + decode 32 full variants + v32 lag2/s2)**:
   read the judge profile's per-variant kernel durations (`d1*`/`v1*` `_gd_full` / `_attn_split` / `hfloor`; `d32a`,
   `d32s3`, `d32p768s3`, `d32p1536`, `d32b16s4` `_gd_full` vs `d32fl2`/`d32fl4`/`d32dense` hfloor; `v32*`), all after an
   explicit flush. Port any decode 1 / verify 1 full variant beating `d1a` / `v1a` by > 0.3 us, and any decode 32 full
   variant beating `d32a` by > 2 us (then also try it at decode 16). H100 CUPTI cold: d32 241.7 / 243.4 / 244.6 / 246.6 /
   248.3, floors random 234.2, dense 209.4. Also compare main's unflushed `_gd_full` with `d1a_gd_full` (s115: warm).
3. agent2's own queue (STATE-agent2.md): agent2-exp/half 52b41bd (nvs cl, verify 32 sliding) is agent2's first check.

## Dead-ends index (NOTES.md line refs approximate; grep the session)
- s125: dynamic tail as built (exp/dtail, +5..+18 us), static gbal range weights by program id (exp/wramp: +1.5..+24 us; the tail is per-replay random).
- s124: extra graph nodes (none in any shape), cluster launch-rate spread (<= 0.1 us, tools/s124).
- s122: "MUFU and tensor serialize in HW" (they overlap; tools/s122/tm.cu).
- s121: nvs (native verify-sliding, NV_HALF/0, S 32-128, NS 5-7) at verify 1 full: cold +0.9..+6 us (exp/nvsf).
- s120: forcing 1 CTA/SM for csl (already 1/SM at batch 1), vtk placement (116 SMs, max 3/SM; fixes = dead layouts).
- s119: decode 1 full stages 3/4, BN 64 s2, BN 16 s4, splits 255/1023 and verify 1 full stages 10/12 (H100; H200 via
  hdiag4), `_gd_full` work stealing (on paper: late grab = 3 trips > 0.86 us spread), csl owner-merge sig/tree (tie again).
- s118: FAST _red_fin (reorder), decode 1 full split counts 396-1300 (all-keyed splits +1.4..+2.5: new token must be in an
  empty program), verify 1 full _combine cw 8/16, _combine_bal cw/hb at verify 16 full.
- s116: decode 1 full end-of-grid atomic burst (whole atomic removal -0.3..-0.4 us cold H100; pre-sum ideas capped there).
- s115: verify 1 full plan knobs (warps 8, BN 128 st 6, BN 32 st 12, splits 63/255, hpp 2) cold and warm.
- s114: physical-order decode full: atomic-free multi-split inversion (18 us H100), windowed L2 sweep with prefetch hints
  (2x DRAM traffic), window-sorted order alone (-3% H100, ~-1.5% H200).
- s113: FPIPE (hand-pipelined fixed-reference loop in Triton _attn_split: no wgmma overlap, +8 barriers/tile).
- s111: native pipelined verify-full (exp/nvf v1-v5): C7512 at 104 regs, 1 CTA/SM x 2 programs starves the V widen.
- s110: pipelined LZ 10 loop in Gluon (C7512 at 2 CTAs/SM; 1 CTA/SM +24 us), ones-column (n72) row sums (ceiling 0).
- s109: csl keys per warp 16 (w8) / 128 (w1, no premerge) vs 32; gw / gbal Gluon at verify 1 full even with LZ 10.
- s107: FOLD (softmax scale/reference subtraction folded into the QK wgmma: extra k-step lengthens the chain).
- s106: fixed reference in _gd_full (decode full gather-bound), at verify 16 sliding (+0.59), verify-full plan knobs under
  LZ 10, ones-column row sums (Gluon needs power-of-two N).
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

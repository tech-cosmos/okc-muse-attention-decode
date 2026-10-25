# STATE-agent2 (short picture; NOTES-agent2.md is the log)

Updated 2026-10-25 ~16:50 UTC (agent2 session 21; s21 = nvs cl direct pushes, H100 verify 32 -0.13..-0.30; s20 = nds S probes dead). Sandbox deleted ~17:45Z: everything pushed (agent2, agent2-exp/*).

## Scope

Batch 16/32 sliding layers only (AGENT2.md). No scored runs; wins go to COORD "Ready to merge".
Checks: <= 8/day for agent2, leave >= 4 of the account's 20 for agent 1.

## Shapes (H200 scored 8aec5993 = main e8a1f73; H100 bench in parentheses)

| shape | H200 us | H100 us | per-us value | path |
|---|---|---|---|---|
| decode 16 sliding | 8.19 | 9.60 | 0.42 %/us | native csl (16-CTA cluster/request) |
| decode 32 sliding | 13.97 (nds: **12.20**) | 15.17 (nds 13.7) | 0.23 | Triton; agent2 652f126: nds + _combine_hb |
| verify 16 sliding | 12.75 (nvs: **12.06**) | 14.27 (nvs 13.5) | 0.31 | Triton; agent2 652f126: nvs S 8 + _combine_hb |
| verify 32 sliding | 17.72 | 18.96 | 0.18 | Triton (256 progs, 8 x 64-key tiles) + _combine_hb |

## Candidates

- (s21) **FIRST CHECK after 00:00Z 2026-10-26: agent2-exp/cldir d5e1482** (or its tip) = halfpfkr abc5f38 + nvs cl direct
  pushes (verify 32 sliding: H100 paired -0.30 (11/11), -0.13 (11/11) vs halfpfkr). tests 16/16. Compare per shape as below.
- (s19) superseded by cldir: agent2-exp/halfpfkr abc5f38 = halfpf 6a9a26a (below) + nds K-early
  barrier (agent2 565c536: H100 decode 32 -0.14 (13/15) on halfpf). tests 16/16. Same per-shape comparison as below.
- (s17) superseded by halfpfkr: agent2-exp/halfpf 6a9a26a = half e5f65ed (verify 32 cl, below) +
  pf 946a445 (page-table L2 prefetch in csl/nds: H100 decode 16 -0.11..-0.13, decode 32 -0.06..-0.24). tests 16/16.
  Compare verify 32 sliding vs 15.4, decode 16 vs 8.13, decode 32 vs 12.20 H200; list only winners.

- (s15) LISTED in COORD Ready to merge: **agent2-exp/nvhalf 7ba35f9** (= agent2 now) = origin/main d93a80d + NV_HALF;
  agent 1's check 2d0bfdc6: verify 16 sliding 11.63 H200 (was 11.90) WIN, verify 32 tie.
- (s15) inside halfpf: agent2-exp/half e5f65ed = NV_HALF + cl at verify 32 (2-CTA DSMEM merge, no combine; H100
  -0.29..-0.86 vs NV_HALF+combine) + split start cluster barrier (H100 -0.19..-0.29 more). tests 16/16.
- Checks used by agent2 on 2026-10-25: 5; account 20/20 (s16-s19: 0).

## Ranked ideas

000000. (s14) DSMEM bulk exchange removes the merge launch where S 2 fits (verify 32 done); look for more S-2 places.
00000. (s13) built as NV_HALF (above). Next: same half split for nds (decode 32), est. <= 0.3 us.
0000. (s11) merges are the wall (~2 us on the H100 for every split design: PDL release ~0.5 + post-wait partial
    reads ~1.2; last-CTA tails the same). Only csl-style small-CTA clusters hide it (decode 16). New structural ideas
    needed; nds NS 4 at decode 16 = kernel 8.6-8.9 (-1.2 vs NS 3) if a cheap merge ever appears.

0000. (s8-s9) nvs (verify) is consumer-bound; consumer trims exhausted (BN2, lazy rescale, QK prefetch dead).

## Dead ends (agent2; agent 1's are in STATE.md / NOTES.md)
- s20: nds S 4 NS 6 tie (+0.05), S 4 NS 8 +0.32, S 6 NS 3 noise, S 5/6 NS 4 +3.5/+4.4 (1 CTA/SM). NOTES-agent2 s20.
- s19: K-early (readyv) in nvs +0.10/+0.19 verify 16/32; nds K and V in separate cp.async groups +0.24. NOTES-agent2 s19.
- s18: nds cos/sin trip (constant-position upper bound -0.13), split-major CTA order +0.35, even/odd split halves +0.11. NOTES-agent2 s18.
- s17: page-table L2 prefetch in nvs (verify 16/32) +0.18/+0.19 (thread 32), +0.32/+0.35 (thread 160). NOTES-agent2 s17.
- s16: nds split pairs packed in one 512-thread CTA + smem merge (agent2-exp/ndp): +0.74 decode 32 (packaging +0.96,
  merge -0.3; loop ~0.5 us slower per split, cause not found). tools/a2/nds_pair_cta_dead.patch. NOTES-agent2 s16.
- s15: cl 2 at verify 16 (split pairs in clusters + global per-warp ticket) +2.59 (sync chain ~3 L2 trips);
  nds pair merge (2-CTA DSMEM, partials halved) +1.06 (start cluster barrier ~0.45 of it). NOTES-agent2 s15.
- s14: cl at verify 16 (S 2, 64 CTAs) +1.86; cl push by st.shared::cluster + per-thread/per-warp release arrives
  (+1.3 / +0.46 vs bulk); IEEE div in epilogues (~1.4 us at 128 threads: use __fdividef). NOTES-agent2 s14.
- s12: nds at decode 16 (+2.1..+2.5 vs csl), nds decode 32 NS 4 (+3.4, 1 CTA/SM) / S 16 (+3.1) / NS 2 (deadlock). NOTES-agent2 s12.
- s11: nds fused last-CTA merge (tools/a2/nds_fm_dead.patch; best 14.05-14.19 vs 13.9 with fast math + acq_rel ticket
  + gate L2 prefetch), native PDL combine (nds_ncombine_tie.patch: tie at 4 d/thread), nds at decode 16 (NS 4 10.8 total;
  CL 8/16 15.3-16.7), vtk at verify 16 (42 us). NOTES-agent2 s11.
- s10: nds in-cluster merge (CL 4 pull 16.5 / push 15.5 vs 14.2 us; CL 8 ~20; cluster launch alone +0.9), nds LAG 2
  (NS 3 deadlock, NS 4 tie), NS 6, combine HB/warps, nds for decode 16 (10.4-11.0 vs csl 9.3). NOTES-agent2 s10.
- s9: nvs BN2 (128-key consumer steps, tools/a2/nvs_bn2_dead.patch; NS 4 deadlocks, needs NS >= 6 and CREGS >= 208)
  +1.1/+1.0 us verify 32/16 kernel; lazy o-rescale (offset 8, T 0.75 or exact T 0) +0.1..+0.3. NOTES-agent2 s9.
- s8: nvs QK(i+1) prefetch before softmax(i) +0.54 us verify 32; partial bulk stores (cp.async.bulk 4 KB/warp) +0.38;
  store cache hints noise; ptall (whole page-table slice by ldg) +0.3/+0.7; cp.async.bulk 128-B row gathers slower
  than cp16; no-merge (request, KV head) verify grid consumer-bound before building. tools/a2/nvs_bst_qkpf_dead.patch.
- s7: no-split (1 CTA per request) batched sliding: gather alone capped at ~11.7 us B32 / 8.3 B16 H100; dnm (mma.sync,
  tools/a2/dnm.cu) 18.6/17.4 us decode 32/16 (FP8 mma.sync is emulated on sm_90: ~15 us of compute). nvs at S 1 ~30 us.
  nvs NV_PW 2 (2 producer WGs) tie. NOTES-agent2 s7.
- s6: nvs distributed in-kernel merge (each split owns 16 rows/head, ticket barrier, tools/a2/nvs_dmerge_dead.patch)
  +1.7 us verify 32 H100 (spin +2.5 after the slowest CTA, L2 row loads +1.5); FMA-poly exp2 offload not built: all
  MUFU exp removed is only -0.2..-0.3 us. H100 verify 32 decomposition: combine 2.8, compute 1.0, V transpose 0.5,
  gather structure ~2.3 above the bare-gather floor (NOTES-agent2 s6). Tool: tools/a2/vab.py (paired variant A/B).
- s5: nvs 2 softmax chains per KV head (4 consumer WGs, tools/a2/nvs_c2_dead.patch): verify 16/32 +1.5/+1.7 us H100
  (0/9); nvs is gather-trip-bound, not consumer-bound. Flag-spin merges closed by agent 1 (NOTES.md s34, exp/spin).
- s4: nvs fused last-CTA merge +1.6..+3.6 us (tail latency-bound on 4 warps/SM); 4-CTA clusters for nvs don't fit one
  wave (30 max); _combine_hb HB/warps other than (2,1); csl 8-CTA clusters (+0.48), csl 64-reg cap (+2.0), csl 2 warps
  x 2 chunks (+2.9). NOTES-agent2 s4.
- s1: verify sliding fuse / dual / hpp 2 / stages 4 / splits 8; decode 32 fuse, block_n 16/64, warps 1/4,
  splits 4/16, hpp 2 (plateau); preq for decode 32 and verify 32.
- s2: in-grid merge via S-CTA clusters + barrier.cluster (`cl`, agent2-exp/clmerge): verify 32/16, decode 32
  +1.87/+1.30/+1.22 us (cluster-4 launch uses 124 SMs, 8 with 3 CTAs: +2.5 us alone). Gluon _gw_verify at
  splits 2-16 on a ctx-2048 proxy: 15.6/19.4 us vs Triton sliding 13.66/18.46 (tools/a2/proxy2.py).

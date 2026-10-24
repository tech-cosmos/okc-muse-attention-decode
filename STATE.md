# STATE (short current picture; NOTES.md is the long log: grep it)

Updated 2026-10-24 ~19:20 UTC (session 70 = unattended session 2 of the #1-push brief).

## Board and best run

- Board 2026-10-24 18:05Z: #1 colinlikescode 2.834x, #2 2.726x, #3 2.624x, #12 2.343x; **ours #14 2.343x** (18:45Z)
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

- main = 14355d4 kernel (+ notes/tools commits). Scored. No unchecked kernel changes on main.
- Dispatch: verify full B*Hkv>4 -> Gluon `_gw_verify` gbal (warp-specialized: 1 producer WG gathers + converts V
  to FP16, 2 consumer WGs = one per KV head, 2 CTAs/SM); decode full -> Gluon `_gd_full`; sliding + small -> Triton
  `_attn_split` + PDL merges.

## Budget (UTC day 2026-10-24; API: `okc problems --json`)

- Account: checks 10/20 used (10 left), scored 9/10 used (1 left). Claude share: 5/8 checks used -> 3 left today;
  scored: only 1 left on the account.

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

1. **Native verify-full kernel with FP8 P@V** (NOTES session 70): native consumer-only microbench (tools/s70/cmicro8)
   152 us with E4M3 P (FA3 Max_offset 8; V^T FP8 K-major with permuted keys, no register shuffles), 149 with lazy
   rescale, vs Gluon consumer-only 178 and native FP16-P 166. Estimated verify 32/16 full H200 210/109 -> ~180/95,
   ~+3.5-4% overall. Risk: the producer's FP8 V transpose (do it FA3's way, ldmatrix.trans + PRMT + stmatrix, or a
   conflict-free 4x4-byte register transpose). Step 1 = producer-only microbench with real gathers + transpose.
   FA3 source citations for E4M3 P are in NOTES session 70.
2. Sliding shapes: no cheap lever left in Triton (cluster merges dead, s70); a native sliding kernel would have to
   beat `_attn_split` + PDL merge on the H200 (Codex's native sliding didn't).
3. Merge tail on verify full (~7 us on the H200 at verify 16/32 = ~0.8% overall at most).

## Queued for checks

- Nothing yet (main is the scored kernel). Use today's 3 Claude checks only on a candidate that wins on the H100.

## Dead-ends index (NOTES.md line refs approximate; grep the session)

- Verify-full Gluon consumer: FA3 pipelining (C7514 serialization, s37), BN 32 (s56), conditional rescale (s18
  exp/wcr), FP16 scores / f16x2 ex2 (s12, s49), FP8 P@V ceiling only -10 us all-L2 (s22), 6 WGs/SM (s42, s51),
  K-ready barrier (s47), V ring split (s46), bank-conflict fix (s45).
- Decode full: near gather floor (s39: dense read 232 vs random 236 us H100).
- Batch-1 sliding: fused last-program merge (s36, s43, s50), Gluon cluster merge (s25: Gluon DSMEM
  codegen was slow; the native push-merge in tools/s69/cmerge.cu is a different, faster pattern).
- Thread-block cluster (DSMEM) merges launched from Triton (s70, exp/cluster): batch-1 sliding +1.7 us, batched
  sliding +3..+6 (cluster launch alone +1.3/+2.3), decode 1 full merge +0.7..+3.2. Dead from Triton/Gluon.
- H100-only sliding wins that lost on the H200: batched sliding hpp 2 (scored 56fca423), Gluon sliding (f3e099c1).

# STATE-agent2 (short picture; NOTES-agent2.md is the log)

Updated 2026-10-25 ~10:30 UTC (agent2 session 9).

## Scope

Batch 16/32 sliding layers only (AGENT2.md). No scored runs; wins go to COORD "Ready to merge".
Checks: <= 8/day for agent2, leave >= 4 of the account's 20 for agent 1.

## Shapes (H200 scored 8aec5993 = main e8a1f73; H100 bench in parentheses)

| shape | H200 us | H100 us | per-us value | path |
|---|---|---|---|---|
| decode 16 sliding | 8.19 | 9.60 | 0.42 %/us | native csl (16-CTA cluster/request) |
| decode 32 sliding | 13.97 | 15.17 | 0.23 | Triton _attn_split (512 progs) + _combine_hb |
| verify 16 sliding | 12.75 | 14.27 | 0.31 | Triton (256 progs, 4 x 64-key tiles) + _combine_hb |
| verify 32 sliding | 17.72 | 18.96 | 0.18 | Triton (256 progs, 8 x 64-key tiles) + _combine_hb |

## Candidates

- agent2-exp/maskif 6c6a87a: MIF (score masks only on tiles that need them) for verify 16/32 sliding + PREQ 1 for
  verify 16. H100 paired: verify 32 -0.61 us, verify 16 -0.37. H200 check a27e9a52 2.376x: verify 32/16 sliding
  16.64/12.49 us. In COORD "Ready to merge" (merged into agent2).

- agent2-exp/nvs fd7fa2e (s3): native verify-sliding kernel `nvs` (tools/a2/nvs_clean.cu embedded as _NVS_SRC) for
  verify sliding batch >= 32: H100 paired verify 32 -1.66 us (9/9), tests 16/16, chk_nvs 5/5. H200 checks 9a841c48 / baea7057: verify 32 sliding
  16.64 -> 15.28 / 15.07 us. Verify 16 on nvs (S 8) tied on the H200 (12.63 both): off. In COORD "Ready to merge".
- Checks used by agent2 on 2026-10-25: 4 (a27e9a52, 9a841c48, baea7057, 058b8dce); s5-s9 used none (account's last 4 are agent 1's).
- (s4) decode 32 sliding on csl (agent2-exp/csl32 80fabbc): H100 -1.02 us but H200 check 058b8dce 13.74 = tie with
  Triton (13.72-14.09): NOT listed. (LAG 2: now in lagS, s9).

- (s9) agent2-exp/lagS (merged into agent2): nvs LAG 2 at S 4 + verify 16 on nvs (LAG 1, S 8). H100 paired: verify 32
  -0.18 (9/9), verify 16 -0.26 (8/9); tests 16/16. NOT H200-checked: queue for the next check with spare budget.

## Ranked ideas

0000. (s9) nvs consumer trims are exhausted: BN 128 steps (+1.1..+1.3 us, chain 20 % slower per key, same instruction
    count) and lazy rescale with offset 8 (+0.1..+0.3) both dead. Next real lever: decode 32 native kernel with both KV
    heads in one m64 (NOTES-agent2 s9 Next 2), priced by prototype first.

000. (s8) nvs is CONSUMER-bound: with no gathers at all it takes 12.2 us at S 4 vs 13.8 (verify 32 H100); the chain
    is ~0.85 us per 64-key tile per consumer WG (ncu: softmax issue + latency). s7's "producer overhead" was the
    consumers + 4 MB of FP32 partial stores (~0.1 us/MB). Lever: fewer instructions per score in the consumer
    (BN 128 steps, transpose off the consumer SMSPs). QK prefetch, bulk stores, no-merge grid: dead (NOTES-agent2 s8).
1. (built s3, see above) Native verify-sliding kernel `nvs` (NO clusters: s2 showed 4-CTA cluster placement alone +2.5 us) (2 consumer WGs per CTA, full-row cp.async gathers, wgmma; design in
   NOTES-agent2 s1 "Next ideas" 2). Biggest lever on verify 16/32 (~-1.7/-3.4 us H100 ceiling).
1b. nvs follow-ups: the ~3.6 us prologue (seq -> page table -> first gathers) is 3 DRAM trips; steady state ~1 us
   per 16-KB tile per SM. s4: fused merges dead, _combine_hb shape already best, LAG 2 -0.18 (verify 32). Small levers
   only: direct page-table loads into producer registers, LAG 2.
2. Decode 32 sliding: smem-staged csl-style 8-CTA clusters (risky; s73 csl8 lost).

## Dead ends (agent2; agent 1's are in STATE.md / NOTES.md)

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

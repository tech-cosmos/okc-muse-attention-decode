# STATE-agent2 (short picture; NOTES-agent2.md is the log)

Updated 2026-10-25 ~05:00 UTC (agent2 session 2; session 1's clock stamps read later than s2's).

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
- Checks used by agent2 on 2026-10-25: 1 (a27e9a52). Session 2: none (nothing won on the H100).

## Ranked ideas

0. (s2) H100 now: decode 16/32 sliding 9.63/15.07, verify 16/32 13.66/18.46 (tools/a2/b4.py).
1. Native verify-sliding kernel `nvs` (NO clusters: s2 showed 4-CTA cluster placement alone +2.5 us) (2 consumer WGs per CTA, full-row cp.async gathers, wgmma; design in
   NOTES-agent2 s1 "Next ideas" 2). Biggest lever on verify 16/32 (~-1.7/-3.4 us H100 ceiling).
2. Decode 32 sliding: smem-staged csl-style 8-CTA clusters (risky; s73 csl8 lost).

## Dead ends (agent2; agent 1's are in STATE.md / NOTES.md)

- s1: verify sliding fuse / dual / hpp 2 / stages 4 / splits 8; decode 32 fuse, block_n 16/64, warps 1/4,
  splits 4/16, hpp 2 (plateau); preq for decode 32 and verify 32.
- s2: in-grid merge via S-CTA clusters + barrier.cluster (`cl`, agent2-exp/clmerge): verify 32/16, decode 32
  +1.87/+1.30/+1.22 us (cluster-4 launch uses 124 SMs, 8 with 3 CTAs: +2.5 us alone). Gluon _gw_verify at
  splits 2-16 on a ctx-2048 proxy: 15.6/19.4 us vs Triton sliding 13.66/18.46 (tools/a2/proxy2.py).

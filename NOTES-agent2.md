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

## Session 2 (2026-10-25 04:41 - ~05:00 UTC by the sandbox clock), branch agent2-exp/clmerge (dead, pushed)

Merged origin/main (e85e702; submission.py unchanged). H100 baseline (tools/a2/b4.py, new: times the 4 shapes):
decode 16/32 sliding 9.63/15.07, verify 16/32 sliding 13.66/18.46 us. No checks used.

### Dead: merge inside _attn_split via thread-block clusters (plan `cl`, agent2-exp/clmerge 5d94cbf, off)

Idea: launch the S splits of each (request, KV head) as one S-CTA cluster, and after the partial stores run
`barrier.cluster.arrive.release` / `wait.acquire` (hardware, co-scheduled, unlike s17's sfuse atomic + spin), then
each CTA merges its MG / S rows from L2 (.cg) with RMSNorm * sigmoid(gate): no _combine_hb launch. How to launch a
Triton kernel as clusters without num_ctas layouts: compile normally (program_id = %ctaid with num_ctas 1), then set
the CompiledKernel's `packed_metadata` = (num_warps, cl, shared): the launcher sets gridDimX = gridX * cl and the
cluster attribute (`_launch_cl` in the branch; only the CL=True specialization is patched). Correct (chk.py 12/12).
H100 paired (abp.py, 9 rounds): **verify 32 +1.87, verify 16 +1.30, decode 32 +1.22 us (0/9 each)**. Ablations at
verify 32 (env CL_ABL / CL_NOCL / CL_N / CL_SMEM in the branch):
- attention alone, no merge, no cluster: 15.54 (so the merge costs 2.6 us of the 18.2 whole: the target).
- same kernel as 2-CTA clusters 15.86 (+0.3), **4-CTA clusters 18.0 (+2.5)**: %smid shows the cluster launch puts
  the 256 CTAs on only 124 SMs (8 idle, 8 SMs with 3 CTAs); a normal launch uses all 132 (124 x 2 + 8 x 1).
  cuOccupancyMaxActiveClusters: 528/264/124/62 for cluster 1/2/4/8. Padding dynamic smem to force <= 2 CTAs/SM:
  23.1 (clusters no longer all resident); 3/SM: 18.0.
- + cluster barrier +0.6, + the slice merge another +1.5. So even a free co-scheduling barrier would leave at most
  ~0.5 us. Don't retry in-grid merges for batched sliding (sfuse s17, fuse s1, cl s2 all lose).

### Dead: Gluon _gw_verify with sliding-sized splits (tools/a2/proxy2.py)

s75's proxy (full layer, ctx 2048, q_len 8) ran gw_verify at its one-wave splits (2-tile programs). Re-run with
splits 2/4/8/16, gfuse on/off, uniform lengths: best B16 15.60 (s4, no gfuse), B32 19.41 (s8) us, before RoPE and
window masks, vs real Triton verify 16/32 sliding 13.66/18.46. The warp-specialized structure loses at this size
even tuned. (The Triton plan on that full-layer proxy reads 31/53 us; that proxy path is not representative.)

### Next ideas (unchanged ranking otherwise)

1. A native verify-sliding consumer still needs >= ~1.6 tiles/us/SM more than Triton at Triton's fixed cost; both
   native attempts with clusters (s72 csv, s84 vsl) and both Gluon structures lose. If anyone builds `nvs`, launch it
   WITHOUT clusters (above: 4-CTA cluster placement alone costs 2.5 us at 256 CTAs) and keep _combine_hb + PDL.
2. Decode 32 sliding: no cluster designs either (same placement issue at 8-CTA clusters: 62 max active).

## Session 3 (2026-10-25 04:52 - ~06:00 UTC), branch agent2-exp/nvs: native verify-sliding kernel `nvs` (WIN on H100)

Merged origin/main (ad2116c: agent 1 merged maskif and submitted). Account at start: checks 10/20, scored 4/10.

### Diagnosis of Triton verify 32 sliding (tools/a2/stamp.py, ao.py, top.py)

- `tools/a2/ao.py`: whole vs attention-only (merge launch no-op'ed): verify 32 18.1 / 15.95 (merge 2.1-2.4),
  verify 16 13.4 / 11.5. Splits 8 attn-only 18.1, splits 2 18.3: ~16 tile-units per SM either way.
- ncu: issue slots 21%, DRAM 41%, SM active 68% of elapsed; stalls spread (long_sb 24%, wait 17%, short_sb 13%,
  barrier 12%). %globaltimer stamps per program (stamp.py, inline asm in a patched copy): durations 11.7-16.5 us;
  1-CTA SMs 11.8 vs 2-CTA SMs 14.2; split 0 (new tokens + window mask) +1.3 us median, but removing _new only moved
  the work to the other CTAs on the SM (kernel max -0.6, graph time flat). Latency-chain bound, not issue bound.

### nvs (tools/a2/nvs.cu = experiment file with ablation knobs; nvs_clean.cu = what ships; nvs.py check/out/time/stamp)

From tools/s71/nverify.cu (agent 1's verify-full native kernel): one CTA per (request, split) of 384 threads, no
clusters, S = 4 (128 CTAs at batch 32 = one per SM, __launch_bounds__(384, 1), setmaxnreg 40/160, NS 4 stages).
Producer WG: page-table entries staged 4 tiles ahead via 4-byte cp.async (the window start is not 16-B aligned),
8 lanes per 128-B row (both KV heads), FA3 ldmatrix.trans V transpose. Consumers (one per KV head): RMS norm + RoPE
+ E4M3 queries into smem, QK(i) || PV(i-1) wgmma pipeline, window mask on the first tile / tail mask by column
thresholds, P in E4M3 with FA3's offset 8 (NV_LAZY 0: the lazy max is NOT used, per NOTES s70's precision note).
New tokens: the last split's consumers normalize/RoPE/quantize the 8 new K/V rows, store them in the cache and lay
them out as a K tile (rows 0-7, zero rows after) and a V^T tile (key slots 0-7 of group 0, zeros elsewhere), then
run them as one causal tensor-core tile BEFORE the cached tiles (the consumers idle ~2 us waiting for the first
gathers anyway), and rescale when the first cached tile's max arrives. Partials in _attn_split's layout -> the
existing _combine_hb (PDL). Dispatch: window 2048, L 8, batch >= 32 (NVS_MIN_BATCH), hq 16, hkv 2, d 64, page 1.

Correctness: partials vs Triton (lse exact, o within E4M3-P rounding); tools/a2/chk_nvs.py 5/5 (batch 32-64, ctx
1500-80000, edges/soft/coherent, kv_scale 1/64), errors equal to Triton's (7.830e-2 vs 7.808e-2 at ctx 1500
edges); tests 16/16 (they never reach nvs: batch <= 5).

H100 (us): verify 32 sliding paired Triton 17.94 -> nvs 16.30 (-1.66, 9/9). Verify 16 at S 8: 13.49 vs 13.47 (tie,
so Triton keeps it).

What mattered / dead ends inside nvs:
- New tokens on CUDA cores after the loop (512 FMA/thread): +2.5 us on the last split (it gates the kernel). In the
  merge kernel before gdc_wait (tools/a2/cnt.py `_combine_nt`): merge 2.9 -> 7.3 us (2048 one-warp programs redo
  the K-new norm/RoPE; they don't overlap the attention). Own CTA per request (S+1 CTAs): +2.4 us (the 32 extra
  CTAs share SMs with main CTAs even when launched last). Early tensor-core tile: ~+0.6 us total. Winner.
- 2 CTAs/SM (S 8, 256 CTAs) 15.5 attn-only vs S 4 14.0. NS 3/4/5/6 flat (4 best by ~0.2), LAG 2 +0.4, L2 prefetch
  of tile i+2 rows +1.3 (i+3: broken/slow), producer-only (consumers just wait) 12.1, consumer chain with stale smem
  (no gathers after NS tiles) 13.5: per tile ~1 us whichever side, so the kernel is a producer<->consumer chain.
- Stamps (nvs.py stamp, NV_STAMP=1): Q ready 2.0 us, first tile ready 4.1 (seq -> page table -> gathers), loop
  ~1.1 us/tile, stores 0.5.
- __launch_bounds__(384, 2) with the new-token code: 80 regs, spills + "wgmma serialized" (C7512): use (384, 1).

### H200 checks (session 3)

- 9a841c48fc014363 (fd7fa2e, nvs at batch >= 32): 2.371x [2.335, 2.406], 16/16. **Verify 32 sliding 15.28 us** (vs 16.64
  in a27e9a52); profile: nvs kernel 10.8 us vs Triton _attn_split 13.2, _combine_hb 2.6 (unchanged).
- baea7057c6d3454c (2b82970, + verify 16 on nvs with 8 splits): 2.381x [2.345, 2.416], 16/16. Verify 32 sliding 15.07,
  **verify 16 sliding 12.63 = same as Triton's 12.63** in 9a841c48: tie, so reverted (NVS_MIN_BATCH 32).
- Branch merged origin/main 9b14e27 (agent 1's csl-per-KV-head, submitted 08bf9391 2.445x); tests 16/16 after the merge.
  Listed in COORD "Ready to merge". Agent2 checks today: 3 (a27e9a52, 9a841c48, baea7057).

### Next ideas

1. nvs verify 32: the prologue is ~4 us before the first cached tile lands (seq -> 4-byte page-table cp.async ->
   gathers). E.g. load seq + page-table entries of the first tiles with plain loads in the producer (no cp.async
   round trip), or prefetch the first tile's rows to L2 from the consumers while they do Q/new tokens.
2. The merge (_combine_hb, 2.6 us H200 at verify 32) is now ~20% of the shape; all in-grid merges lost before.
3. Decode 32 sliding (H200 14.0): an nvs-style decode kernel (rows = 8 per KV head pads wgmma M 64 8x: would need
   both KV heads + several requests per CTA, or mma.sync) - unexplored.

## Session 4 (2026-10-25 05:57 - ~06:55 UTC): csl at decode 32 (H200 tie), nvs fused merge (dead), nvs pipeline probes

Start: origin/main 19c85c6 already in agent2. Account at start: checks 13/20 (agent2 3), scored 5/10.

### decode 32 sliding on agent 1's csl-per-KV-head kernel (branch agent2-exp/csl32 80fabbc, CSL_MAX_BATCH 32)

- H100 paired vs Triton: 15.15 -> 14.24 us (-1.02, 7/7); tools/a2/chk.py 12/12, tests 16/16.
- **H200 check 058b8dcecaca437b: 2.390x [2.354, 2.426], 16/16; decode 32 sliding 13.74 us** vs Triton 13.72-14.09 in the
  last six checks (mean ~13.92): a TIE. Profile: csl 12.7 us one kernel vs Triton _attn_split 10.8 + _combine_hb 1.5.
  csl moves ~1.35 TB/s at both B16 (6.2 us / 8.4 MB) and B32; Triton's attention 1.55 TB/s on the H200. Not listed.
- Variants, all H100 decode 32 (tools/a2/abg.py = paired A/B of a submission global):
  - 8-CTA clusters x 8 warps (tools/a2/csl_cl8.patch): +0.48 us vs 16-CTA (0/9).
  - csl at 79 regs fits 6 CTAs/SM = 792 of 1024 CTAs (two waves). __launch_bounds__(128, 8) -> 64 regs + spills:
    16.19 (+2.0). 2 warps per CTA x two 32-key chunks per warp (tools/a2/csl_kpw2_dead.patch, 135 regs -> 128 cap with
    small spills, 1056 slots): +2.9 us. Fewer warps hurt more than the second wave: csl is warp-latency-bound.

### nvs fused merge (last-arriving CTA merges; tools/a2/nvs_fuse_dead.patch): DEAD

Per (request, KV head) int32 arrival counter (self-resetting), FP32 partials as before, the merging WG does
_combine_hb's work (weights, RMS norm, sigmoid gate, BF16) from L2 + its own registers. Correct (chk_nvs 7/7) but
verify 32 sliding H100: +3.6 us first version, +1.6 us best (split S-1 designated merger that skips its own partial store
and spins on ld.acquire; all 128 CTAs resident). Warm kernel ablation (tools/a2/prof.py, us): nvs alone 10.76; + publish /
wait 11.45; + cp.async of the 3 other partials (48 KB) + gates 12.76; + merge math 13.93; + gate/stores 14.80. The tail
runs on 4 warps per SM (one per scheduler), so each step is latency-bound; _combine_hb spreads the same work over ~15
warps/SM on every SM and its exposure is only ~2.4 us. Also: unrolled per-split L2 loads serialize (6 trips) and the
first version spilled (68 B) when all loads were hoisted. 4-CTA clusters for an in-cluster merge can't fit: max 30
active 4-CTA clusters at 1 CTA/SM (tools/a2/occ.py), nvs needs 32; 2-CTA clusters fit (66) but don't remove the merge.
- _combine_hb shape after nvs (HB, warps): (2,1) best; (1,1) +1.38, (4,1) +0.62, (4,2) +0.58, (8,2) +1.49, (8,4) +0.70, (2,2) +1.15.

### nvs stamps (shipped kernel; tools/a2/st2.py patches %globaltimer into a diag copy)

H100, us from consumer start: verify 32 (S 4, 8 tiles): Q 1.5, first tile ready 3.6 (max 4.6), loop done 12.0, stored
12.3; verify 16 (S 8, 4 tiles): first tile 3.6, loop 7.2, stored 7.7. ~1 us per 16-KB tile per SM = ~2.1 TB/s;
no last-split tail in the shipped kernel (the "+2 us last split" from tools/a2/nvs.py stamp is its old CUDA-core
new-token path; don't use nvs.py stamps). Little's law: LAG 1 keeps 2 tiles (32 KB) in flight per SM.
- NV_LAG / NV_NS (tools/a2/lag.py, lagab.py): verify 32 paired LAG 2 vs 1 **-0.18 us (11/11)**, verify 16 +0.11;
  NS 6 + LAG 2/3 no better. Small; queue for a bundle, not worth a check alone.

### Checks (agent2 today: 4: a27e9a52, 9a841c48, baea7057, 058b8dce)

### Next ideas

1. Batched sliding attention is at ~1.5-2.1 TB/s of random 128-B gathers after a ~3.6 us 3-trip prologue
   (seq -> page table -> rows); the merges (1.5-2.6 us H200) are the other fixed cost, and every in-grid/fused merge has
   lost (s1 fuse, s2 cl, s4 nvs fused). Remaining levers are small: nvs LAG 2 (-0.18 verify 32), direct page-table
   loads in the producer (skip the smem staging + bar), more bytes in flight per SM at the start (prologue).
2. Untested: verify 16 on nvs with LAG 2 + the first tiles' page-table entries loaded straight into registers.

## Session 5 (2026-10-25 06:34 - ~07:10 UTC): nvs with 2 softmax chains per KV head (DEAD)

Start: merged origin/main f2f0ee5 (agent 1 s93: nvs merged on main 4e63331, check 13582a1b 2.465x) into agent2 (cdfc982);
tests 16/16, H100 bench decode 16/32 sliding 9.41/15.02, verify 16/32 sliding 13.76/16.43. Account at start: checks 15/20
(agent2 4), scored 6/10. Only 1 check was within agent2's share (leave >= 4 for agent 1): not used (nothing worth it).

### nvs NV_C2 2 (tools/a2/nvs_c2_dead.patch, A/B tools/a2/c2ab.py)
Idea: s3/s4 ablations (producer-only 12.1, consumer-only with stale smem 13.5, both 14.0 us H100 verify 32) looked like
a consumer<->producer chain at ~1 us per tile, so double the consumers: 5 WGs (producer + 4 consumers), consumer WG
(h, c) takes KV head h's tiles c, c + 2, ...; chain 1 hands o / m / l to chain 0 through smem (named barrier per head),
chain 0 merges and stores the partial (same _combine_hb layout). New tokens only on chain 0. Correct (chk_nvs 7/7).
- Gotcha: with 640 threads ptxas sets the kernel's register count to the launch bound (96), so setmaxnreg.inc 112 with
  dec 40 needs more than the CTA's pool and **hangs**; 104 fits (5120 + 512 x 104 <= 640 x 96). 96 regs, 4 B spill.
- H100 paired (9 rounds): verify 16 sliding 13.23 -> 14.80 (+1.54, 0/9), verify 32 16.64 -> 18.24 (+1.66, 0/9).
  With NS 6 + LAG 2 (7 rounds): +1.68 / +1.20, 0/7 each.
- So nvs is NOT consumer-chain-bound: twice the consumer parallelism makes it slower (more warps share the SM's issue
  slots and the producer WG; plus a second Q prologue per head and the in-CTA merge). With LAG 2/3 and NS 6 also flat
  (s4), the ~1 us per tile is the gather trip itself; verify 32 nvs (10.8 us H200) and decode 32 Triton (10.8 us H200)
  move the same 16.8 MB in the same time, so ~10.8 us is the H200 cost of the 3-trip (seq -> page table -> rows)
  random-row gather at this size, whichever kernel does it.

### Checked as closed before building (NOTES.md)
- Merge kernel that spins on per-request flags instead of griddepcontrol.wait: NOTES.md s34 (flagmerge) and exp/spin
  (s52-ish, line ~1984): release + poll loses to the PDL grid wait (+0.9..+1.7 us). Don't retry.

### Next ideas (small; none worth a check alone)
1. nvs NV_LAG 2 at verify 32 (-0.18 us H100) if a bundle check goes out.
2. A decode-32 nvs (L 1, rows padded to M 64): expected ~= Triton's 10.8 us (same bytes, same trip structure), merge
   4 vs 8 splits maybe -0.3 us. Low value (0.23 %/us).
3. The batched sliding shapes now sit at ~1.8 us fixed harness overhead + ~10.8 (B32) / ~6.5-9.4 (B16) us of gather +
   1.5-2.6 us merge. Every merge-removal path is closed (fused last-CTA, cluster, flag spin). The remaining structural
   lever would be cutting a DRAM trip from the prologue (seq -> page table), which the inputs don't allow for batch > 1.

## Session 6 (2026-10-25 07:03 - ~07:50 UTC): nvs cost decomposition; poly exp2 and distributed fused merge (DEAD)

Start: merged origin/main 75081b1 (agent 1 s94/exp red, batch-1/full only) into agent2 (0ccaeb0). Account checks
16/20 used at start (agent2 4): the 4 left are agent 1's (AGENT2.md: leave >= 4), so 0 checks this session.
New tool tools/a2/vab.py: paired A/B of string-replacement variants of submission.py (`name|old=>new|...`), any shapes.

### Where verify 32 sliding's 16.6 us (H100 wall, nvs S 4) goes (vab.py ablations, 5-9 paired rounds, all n/n)
- ex2 -> identity (no MUFU): -0.16..-0.31 us (verify 16 on nvs: -0.37). So FMA-polynomial exp2 offload (FA4 style;
  degree-5 minimax on [-0.5, 0.5] has max rel err 2.2e-7, = ex2.approx) can win <= ~0.1 us: **not built, dead**.
- consumer compute removed (consumers only wait ready / arrive empty): -1.02. V transpose removed: -0.46.
  Both removed (gathers + Q + new tokens + stores + combine): 15.07-15.42. Deeper rings on that gather-only kernel
  (NS/LAG/PFD 6/3/4, 8/5/6, 8/6/7) all 15.2-15.3: in-flight depth is not the limit. (PFD must be > LAG, else the
  page-table ring is read before its cp.async lands -> illegal address.)
- _combine_hb removed: **-2.78 us** (nvs alone 13.81). Gather-only without combine 12.42 vs the bare gather floor
  ~10.1 (NOTES.md s85, decode 32 bytes).
- So: combine/kernel boundary 2.8, nvs gather structure + prologue ~2.3 above the floor, compute ~1.0, transpose 0.5.

### Distributed in-kernel merge (tools/a2/nvs_dmerge_dead.patch, stamps tools/a2/dmstamp.py): DEAD, +1.7 us
Each of a request's S = 4 CTAs owns rows [16 k, 16 k + 16) per KV head = exactly warp k's wgmma fragment rows: owners
keep them in registers, the rest is stored as FP32 partials; WG bar.sync, owner warp: __threadfence + atomicAdd ticket
(counter stays a multiple of S, no reset), spin on ld.acquire until the next multiple, ldcg the other 3 splits' rows,
merge, RMS-norm, sigmoid gate, BF16 store. Correct (chk_nvs 7/7, fused at batch 32/33; off when S x B > SMs).
- First build: v[7][8] with runtime S -> +6.5 us (verify 32 23.9 vs 17.3); S fixed at 4 -> **+1.7 (18.6 vs 16.9)**;
  nanosleep backoff in the spin: no change (not contention). 160 regs, no spills.
- Stamps (us from start; globaltimer ~0.26 us steps): loop done med 12.0 / max 13.1; spin done 15.6 (= +2.5 after
  the slowest CTA: write acks + fence + atomic + poll); other splits' rows loaded 17.4 (+1.5 for 3 x 4 KB L2 reads per
  warp); stored 18.4. The kernel boundary + _combine_hb does the same job in 2.8 us wall. With s4 (designated
  merger) and s2 (clusters) this closes in-kernel merges for nvs: cross-CTA L2 handoff inside a kernel is slower than
  the kernel boundary on the H100.
- Loop-end spread (~1 us between median and max CTA) is the same for every split (new-token split 3 is not the
  tail): it's random-gather latency variance, nothing to rebalance statically.

### Next ideas (all small; none worth a check alone)
1. nvs LAG 2 at verify 32 (-0.18 H100, s4) in a bundle check.
2. Cut the nvs gather structure's ~2.3 us over the bare-gather floor: the floor issues every row at once from 2048
   blocks; nvs issues 2 tiles per SM at a time and deeper rings don't help, so the difference is probably the prologue
   (seq -> page-table stage via cp.async + bar -> rows) and the per-tile wait/commit structure; try direct
   page-table loads into producer registers for the first NV_PFD tiles (skips cp.async wait + bar.sync + lds).

## Session 7 (2026-10-25 07:44 - ~09:20 UTC): no-split kernels priced (capped); nvs producer is the real overhead

Start: merged origin/main b34de45 (agent 1 s95-s96: decode 1 full atomic merge, vtk verify 1 sliding) into agent2.
Account checks 16/20 used at start: the last 4 are agent 1's (AGENT2.md), so 0 checks this session.

### Gather throughput by CTA count (tools/a2/nm.py; H100, graph-timed, seq -> page table -> 128-B K and V rows)
| layout | decode 16 | decode 32 |
|---|---|---|
| 1 CTA per request (B CTAs), registers, R 8, 1024 thr | 8.2-8.3 | 11.5-11.7 |
| 1 CTA per KV head (64-B halves), 2B CTAs | 9.1 | 12.1 |
| 2 CTAs per KV head reading the same halves (L2 reuse), 4B CTAs | 8.8 | 12.9-13.1 |
| 4 splits x B CTAs (= nvs's grid), registers, R 8, 512 thr | **6.7** | **10.2** |
| 2 splits x B, R 8, 512 thr | 7.3 | 10.4 |
| cp.async HS1, 512 thr x depth 4 (pt staged in smem) | 8.75 | 12.1 |
| cp.async HS1, 128 thr x depth 12 / 6 | 11.8 / 11.7 | 13.5 / 13.2 |
So a no-split (no-merge) kernel at batch 32 can't gather faster than ~11.7 us H100 (~1.6 above the all-SM floor),
and cp.async needs many issuing threads (per-thread outstanding cp.async is the limit, not ring depth).

### dnm (tools/a2/dnm.cu, tools/a2/dnm.py): no-split decode sliding, 1 CTA per request: DEAD as built
8 warps each own 256 window keys and stream 32-key stages (both heads' 128-B K/V rows, 8 KB) through private
cp.async rings, csl's mma.sync E4M3 math + online softmax, in-CTA merge. Correct first try (8 cases incl. page 64,
kv_scale, short ctx). H100: decode 32/16 sliding **18.6/17.4 us** vs main 15.2/9.4. Ablations: compute alone ~15 us,
gather alone 14.1/11.0. SASS: FP8 `mma.sync` on sm_90 is emulated (F2FP e4m3->f16 converts + 2 HMMA each), ~700
instructions per 32-key stage (both heads) -> ~6.6 us of pure issue per SMSP for a 2048-key CTA; with 2 warps per
SMSP the dependent chains stretch that to ~15. (First build with a fully unrolled stage loop + slot[] in registers:
255 regs + stack, 30-71 us.) A no-split decode kernel would need wgmma (native FP8) and a 512-thread gather; best
case ~12.5 us H100 at B32 (-2.7, ~0.6 % overall), likely less on the H200 (per-SM latency, not bandwidth): parked.

### nvs at S 1 (tools/a2/nvs1.py: nvs itself with one CTA per request, kernel only, no combine)
verify 16/32: ~29.5-30 us at S 1 (NS 4..8, LAG 1..5 all equal) vs 11.8/13.9-14.2 at S 4. Ablations at S 1 / S 4
(verify 32): no softmax 24.8 / 12.6, no wgmma 25.2 / 13.5, no V transpose 27.8 / 13.7, no consumer compute 21.5 / 12.6.
=> **nvs's gather-only time at S 4 is 12.6 (B32) / 9.5 (B16) vs 10.2 / 6.7 for a register gather on the same grid**:
the 128-thread cp.async producer costs ~2.4-2.8 us. This (not the consumers) is the lever for verify 16/32.

### nvs with 2 producer warpgroups (NV_PW 2, 512 threads; tools/a2/nvs_pw2_tie.patch): TIE, not kept
Correct (chk_nvs 7/7). H100 paired 9 rounds: verify 32 PW 1 vs PW 2 +0.10 (PW 2 = 16.53); verify 16 on nvs (S 8):
PW 2 13.23 vs PW 1 13.26 (both -0.5 vs Triton's 13.81 on the H100, but the H200 tied there in s3). So halving each
producer thread's cp.async count doesn't close nvs's ~2.4 us gather gap: the cause is elsewhere.

### Next ideas
1. Find nvs's 2.4-2.8 us producer overhead by ablation on the gather-only (nocomp) kernel at S 4: (a) the prologue
   (seq -> page-table cp.async stage -> cp_wait<0> + bar.sync -> rows) vs nm's direct ldg chain; (b) the per-tile
   cp_wait<LAG> + bar.sync 1 + transpose + fence.proxy.async + arrive; (c) cp.async.mbarrier.arrive.noinc completion
   instead of cp_wait + named barrier; (d) register-staged gathers (ldg -> st.shared) like nm, which hit the floor.
   Worth up to ~2.4 us on verify 32 (0.18 %/us) and ~2.8 on verify 16 if nvs then beats Triton there (0.31 %/us).
2. No-split kernels: parked (gather capped at ~11.7 us B32 on 32 SMs; needs a wgmma build for ~0.6 % at best).
3. Same producer lesson may apply to Triton decode 32 (_attn_split 10.8 H200): compare its gather to nm S4 10.2.

## Session 8 (2026-10-25 08:10 - ~09:10 UTC): nvs is consumer-bound, not producer-bound; partial stores priced

Start: merged origin/main 989f43b (agent 1 s97-s98, batch-1 only) into agent2 (6da7cb9). Account: 16/20 checks used,
4 left; human (COORD 08:10Z): agent 1 spends 1 on main, agent2 keeps <= 2. **0 checks used** this session.
Tools: tools/a2/nmb.py (gather mechanisms on nvs's S x B grid), tools/a2/nvs2.py (kernel-only nvs ablations,
stacked string variants, `NVS_S=1,4` env), tools/a2/nh.py (no-merge (request, KV head) grid gather), nvs_one.py (ncu).

### s7's "producer overhead" is not the producer (H100, kernel-only, no combine)
- nmb.py on the S4 / S8 grid: nvs-style cp16 with 128 threads, 2 tiles in flight = 10.46 (verify 32) / 7.63
  (verify 16 S8): the register floor (10.2). Deeper rings, 512 threads, cp.async.mbarrier.arrive.noinc: same.
  cp.async.bulk 128-B rows (1 per thread, complete_tx): worse (11.5 / 8.1; 64 issuing threads 18.4). +256 idle
  threads, +88 KB smem: no change.
- nvs2.py stacked (verify 16 S8 / verify 32 S4): base 10.08 / 13.95; no consumer compute 9.50 / 13.02; + no V
  transpose 9.25 / 12.51; + no Q prologue 8.83 / 12.14; + no consumer waits 8.72 / 12.37; producer only (consumers
  return) **7.38-7.60 / 11.23** = about the floor. The leftover in the stripped kernel was the partial stores:
  stripped without stores 7.49 / 10.86.
- Partial stores (4 MB FP32 at both shapes): full kernel without them 9.41 / 13.12 (-0.7 / -0.8 kernel-only). With
  _combine_hb in the timing it shows as 0 (vab 'nostore' +0.02) because the combine then reads stale partials from
  DRAM instead of L2. nmb.py with a coalesced 32 KB/CTA float4 store: +0.35-0.5 at the CTA's end, +1.0 when issued
  at the start (overlapping the gathers): **~0.1 us per MB of partials, set by the bytes, not the store pattern**.
  Dead: bulk stores (warp's 16 rows staged in its free V^T slot, one 4-KB cp.async.bulk per warp): verify 32 +0.38
  (9/9); __stcg / __stcs / __stwt: noise (-0.08..+0.05). FP32 partials stay (rules).
- **nvs is consumer-bound**: nvs with NO gathers (cp16 disabled, everything else kept) = 29.1 us at S 1 (32 tiles per
  consumer WG) and 12.2 at S 4 vs 13.8 with gathers (verify 32). So the consumer chain is ~0.85-0.9 us per 64-key
  tile per warpgroup and gathers add only ~1.6 us at S 4. ncu (S 1, no gathers): stalls wait 23 %, selected 18 %,
  not_selected 12 %, long_sb 9 %, short_sb 9 %, mio 7 %, barrier 7 %; by opcode MUFU.EX2 18 %, FMNMX 8 %, PRMT 7 %
  (producer transpose), FADD 6 %: half issue-bound, half latency in the softmax.

### Dead before building: no-merge verify sliding (1 CTA per (request, KV head), whole window, in-CTA epilogue)
nh.py: its gather is fine (HS2, 256 cp.async threads, 8 x 8-KB ring: 11.94 verify 32 / 10.82 verify 16 vs nvs's
16.6 / 13.3 totals), but each SM would hold 32 tiles per head on 64 (B32) or 32 (B16) SMs = 2x today's consumer
tiles per SM at ~0.85 us per tile per chain: consumer-bound above today's time unless the per-tile chain gets ~2x
faster (3+ consumer WGs don't fit the register file at 160 regs).

### QK(i + 1) prefetch before softmax(i) (FA3 intra-warpgroup overlap; tools/a2/nvs_bst_qkpf_dead.patch NV_QKPF): DEAD
Second score array, CREGS 192 (128 x 40 + 256 x 192 regs fit), throwaway QK on the last step (a conditional issue
made ptxas serialize every wgmma, C7515). Correct (chk_nvs 7/7, same errors). H100 paired 9 rounds: verify 32 +0.54
(9/9), verify 16 (Triton) noise; NS 5 doesn't recover it. Gather-free S 1: 31.5 vs 29.4 us, so the chain itself
got slower (the QK latency was not what each step waited on). CREGS 192 alone: neutral.

### Side numbers
- verify 16 on nvs (NVS_MIN_BATCH 16, S 8) vs Triton: H100 paired -0.46 (9/9) this session (s3: -0.14; the H200
  tied at 12.63 in check baea7057). Bundle-only candidate, not worth a check alone.

### Next ideas
1. The lever on verify 16/32 sliding is the nvs consumer's per-tile chain (~0.85 us per 64-key tile per WG): cut
   instructions per score without lowering precision: e.g. BN 128 consumer steps (one max/shuffle/alpha/rescale per
   128 keys instead of 64; m64n128 QK); move the V transpose off the SMSPs the consumers use (PRMT 7 %); FMNMX3
   (3-input max, sm_90 has FMNMX only 2-input: check SASS); packed f16x2 for the row sums is NOT allowed (precision).
2. Fewer partial bytes is worth ~0.1 us/MB, but FP32 is mandatory and the split count is set by gather parallelism.

## Session 9 (2026-10-25 09:07 - ~10:30 UTC): nvs consumer trims: BN 128 steps and lazy rescale (both DEAD)

Start: merged origin/main cc93187 (agent 1 s99-s101, batch-1/full only) into agent2 (7ffbdd0). Account: 18/20
checks used, 2 left = agent 1's (AGENT2.md: leave >= 4) -> **0 checks** this session.

### BN2: consumer steps of 2 producer stages (128 keys): DEAD (tools/a2/nvs_bn2_dead.patch, NV_BN2)
Idea (s8 Next 1): one max / shuffle / alpha / o-rescale per 128 keys; the producer is unchanged (64-key stages), the
consumer issues 4 QK wgmmas into sA/sB and 4 PV wgmmas; a missing odd tail tile reuses the first stage and is
masked (every wgmma issues unconditionally). Correct (chk_nvs 7/7).
- NS 4 DEADLOCKS (GPU 100 %, hang): the producer marks tile t+1 ready only after issuing t+2 (LAG 1), which needs
  the slot of t-2 that the consumer holds until both tiles of its next pair are ready. Needs NS >= 6.
- At CREGS 160 ptxas serializes the wgmmas (C7512); CREGS 208/232 + PREGS 24 fix that (no C7512, 16-88 B stack).
- H100 kernel-only (nvs2.py): verify 32 S4 15.1-15.3 vs BN 64 13.9-14.0; verify 16 S8 11.2-11.6 vs 10.1-10.2.
  Gather-free S 1 (consumer chain only): 33.8-34.4 vs 27.9 us. So the 128-key chain is ~20 % SLOWER per key.
- ncu (S 1, no gathers): BN2 executes about the same instructions (3.16 M vs 3.08 M) at 7.0 vs 6.0 cycles per issued
  instruction (eligible warps 0.68 vs 0.83): the per-step fixed work was never the cost; longer steps only hurt
  latency hiding between the two consumer warps on each SMSP.

### Executed opcodes, shipped nvs (BN 64), per consumer warp per 64-key tile (ncu sass__inst_executed_per_opcode)
FADD 38, FFMA 37, FMUL 37 (o rescale), FMNMX 36, IMAD 35, MUFU 35, F2FP 17, LOP3 13, R2UR 12.5, PRMT 9 (producer),
BRA/SHF/ISETP/VIADD ~8 each; ~380 total vs ~145 essential (FADD/FFMA/MUFU/FMNMX 32 each + F2FP 16). The loop SASS
has ~13 IMAD per step (descriptor/address math, mask path), not a pile of register moves.

### Lazy o-rescale with FA3's offset 8 kept: DEAD
Precision argument (why it would be legal): with a stale max at most T below the true max, P <= 2^(8+T); E4M3 max
448 = 2^8.81, so T <= 0.75 never saturates and loses no small-weight range (s78's objection was lazy offset 0).
Built as NV_LAZY 1 + threshold NV_LZT (offset fixed 8). H100 kernel-only, 2 rounds: verify 32 base 14.06-14.22,
T 0.75 14.30-14.34, T 0 (exact: skip only when no row grows) 14.22-14.26; verify 16 S8 base 10.13-10.27, T 0.75
+0.16, T 0 +0.32. The warp votes cost more than the 32 FMULs they save (S4: only 8 tiles per split, maxima still
grow often). Not kept.

### Next ideas
1. The consumer per-tile chain is latency + issue mixed at 2 consumer warps per SMSP; neither longer steps (BN2), more
   chains (s5: 4 WGs) nor lazy rescale helps. Remaining instruction fat (~0.2-0.4 us at verify 32): uniform
   descriptors hoisted out of the loop (R2UR/LOP3/SHF ~35 per tile), mask path only on the first/last tile.
2. A decode-32 native kernel with both KV heads in one m64 (rows 0-7 head 0 queries in d 0-63, rows 8-15 head 1 in
   d 64-127, zeros elsewhere; QK as m64n64k128 over the 128-B K rows; PV n128 and take each head's half): one consumer
   WG per CTA, only warp 0 has live rows, so consumer issue is tiny and the kernel should sit near nvs's producer-only
   time (11.2 H100 kernel) + combine. Ceiling ~-1 us H100 on decode 32 (15.2); csl32 showed H100 wins can tie on
   the H200. Price it with nmb.py-style prototypes first.

### Candidate for the next day's check: agent2-exp/lagS (nvs LAG by split count + verify 16 on nvs)
vab.py (paired, combine included) on agent2: LAG 2 everywhere: verify 32 -0.26 (7/7) but verify 16-on-nvs only -0.16
vs -0.38 for nvs16 at LAG 1. So the kernel is now templated on LAG: LAG 2 at S <= 4 (verify 32), LAG 1 at S 8
(verify 16), and NVS_MIN_BATCH 16. Tests 16/16, chk_nvs 7/7. H100 paired vs agent2 (9 rounds): verify 16 -0.26
(8/9), verify 32 -0.18 (9/9). H200 risk: verify 16 on nvs tied Triton in baea7057 (12.63 both). Check it in a bundle
when the account has checks to spare; keep each half only for the shape where it wins.

## Session 10 (2026-10-25 10:33 - ~11:25 UTC): native decode-sliding kernel nds = H100 WIN on decode 32 (-1.17 us)

Start: merged origin/main a2d846b (agent 1 s102-s106, batch-1/full only) into agent2 (03281cf). Account 18/20 checks
used, 2 left = agent 1's (AGENT2.md: leave >= 4) -> **0 checks** this session.

### nds (tools/a2/nds.cu, harness tools/a2/nds.py; branch agent2-exp/nds 7521672, merged into agent2)
s9 Next 2, built: nvs's producer warpgroup unchanged (both heads' 128-B K/V rows by cp.async, V^T transpose) + ONE
consumer warpgroup for both KV heads: m64 rows 0-7 = head 0's 8 query heads (Q in d 0-63 of a 128-B SW128 row, d
64-127 zero), rows 8-15 = head 1's (d 64-127), rows 16-63 zero. QK = 4 x m64n64k32 over the whole K row, PV =
m64n128 over both heads' V^T (the two per-head SW64 V^T tiles are contiguous = one 128-row B operand); row r keeps its
head's 64 output columns. Only warp 0 runs mask/softmax/rescale (warps 1-3 only join the wgmmas with P = 0). The
step's new token (both heads, warp 0 lanes) is RMS-normed/RoPE'd/E4M3'd, written to the cache and attended as an
early 1-key tile on the last split (as nvs). Partials [B, 2, S, 8, 64] FP32 -> _combine_hb (HB 2, 1 warp).
Correct first build (nds.py check: 8 cases x S 4/8, errors equal to the Triton path's; kernel_fn 6 more cases).
H100 (nds.py time, k = kernel only):
| config | decode 32 total | kernel |
|---|---|---|
| main (Triton _attn_split 512 progs + combine) | 15.0-15.3 | - |
| NS 4, 1 CTA/SM, S 4 / S 8 | 14.3 / 17.2 | 12.4 / 15.0 |
| NS 3, MINB 2 (2 CTAs/SM, 128 launch regs), S 4 / S 8 | 14.2-15.2 / 14.5-14.8 | 12.9 / 11.9 |
| NS 3, __maxnreg__ 96 (PREGS 40 / CREGS 152), S 4 / S 8 | 14.1-14.2 / 14.0-14.2 | 12.8 / 12.0 |
At MINB 2 the CTAs fill the register file, so the PDL combine can't become resident early (+0.8 us combine at S 8);
__maxnreg__ 96 fixes that (ptxas needs >= 94 for the n128 wgmma). LAG 2: deadlocks at NS 3 (producer needs the slot
the consumer holds), no gain at NS 4; NS 6 no gain. Combine shape (HB 1/2/4/8, 1-2 warps): noise.
**Paired vab (submission, 9 rounds): decode 32 sliding 15.10 -> 13.94 us (-1.17, 9/9)**; S 4 instead of 8 +0.21
(1/7). Tests 16/16. Decode 16 on nds: 10.4-11.0 (S 8/16) vs csl 9.3: off (NDS_MIN_BATCH 32).
H200 risk: csl32 (s4) won -1.0 on the H100 and tied on the H200; nds is a different structure (gathers from 256
CTAs on 132 SMs like nvs, which held its H100 win on the H200). Needs a check.

### In-cluster merge for nds (tools/a2/nds_cl_dead.cu, NDS_CL): DEAD
One S-CTA cluster per request, partials in smem, merge + RMS-norm + gate in-kernel (no combine launch). Correct.
CL 4 (2 CTAs/SM): pull merge (ld.shared::cluster) 16.5 us; push merge (st.shared::cluster to the row's owner, one
barrier) 15.5; vs 14.2 non-cluster + combine. Ablation (pull): cluster launch alone +0.9 us over the same kernel
unclustered, 2 cluster barriers +0.9, remote-load merge +1.7. CL 8: 19.8-21.5 (placement). Clusters stay dead (s2).

### Next ideas
1. The combine is now ~2 us of nds's 14 (kernel 12.0 vs gather floor ~10.2): a merge that doesn't wait for a
   separate launch, without clusters: e.g. per-request ticket + last-CTA merge (dead for nvs at 64 rows/head, but
   nds partials are 8x smaller: 32 KB per request at S 8). Price on the H100 against the 2 us.
2. nds kernel itself: 12.0 vs 10.2 floor: try PFD / staging tweaks as on nvs (small).

## Session 11 (2026-10-25 11:23 - ~12:10 UTC): nds merge alternatives (all DEAD); nds at decode 16 re-priced

Start: merged origin/main (agent 1 s106-s108: batch-1/full only) into agent2. Account 18/20 checks used, 2 left =
agent 1's (AGENT2.md: leave >= 4) -> **0 checks**. Branch agent2-exp/ndsfm (notes + dead patch only).
All H100, tools/a2/nds.py time, decode 32 sliding, S 8, NS 3, __maxnreg__ 96 (= the agent2 nds config); "total" =
nds + merge, "kernel" = nds alone. Reference this session: main (agent2 kernel_fn, nds + _combine_hb) 13.6-14.0,
harness nds + _combine_hb 13.86-14.08, nds kernel alone 11.94-12.19 (combine exposure ~1.9 us).
(Harness gotcha: `CNT.setdefault(B, torch.zeros(...))` evaluates torch.zeros every call = an extra memset node in the
graph, +1 us. Build such buffers once.)

### Fused last-CTA merge in nds (NDS_FM, tools/a2/nds_fm_dead.patch on tools/a2/nds.cu; nds.py needs a trailing
### int32 [B] counter tensor arg, built once): DEAD, best +0.2..+0.4 us
Per-request int32 ticket (self-resetting), last split's consumer WG (128 threads, thread = query head x 8 d, all 8
splits' loads issued at once, no spills) does _combine_hb's math + RMS norm + gate. Correct (nds.py check 8/8).
| variant | total |
|---|---|
| first build (threadfence + atomicAdd, merger threadfence, IEEE div/expf) | 15.20 |
| + gate row prefetch.global.L2 at consumer start (the gate input is cold) | 14.93 |
| + atom.add.acq_rel.gpu, no writer/merger __threadfence (bar.sync carries the acquire) | 14.66-14.78 |
| + fast math (__frcp_rn, __fdividef, ex2 sigmoid, * 1/HD) | **14.05-14.19** |
| gate cp.async'd to smem at start instead of the L2 prefetch | 14.35-14.48 |
Ablations (kernel): ticket only, no merge 12.8 (+0.7 over 12.1; fences removed doesn't change it); merge math with
no partial loads 13.4-13.7; partial loads ~+0.6. Lesson (as nvs s4): a 4-warp tail is latency-bound per dependent
instruction; IEEE div/expf alone cost ~0.6 us. Floor of this design ~ kernel + 2 us = the separate combine.

### Native PDL combine (ndsc, tools/a2/nds_ncombine_tie.patch: cudaLaunchKernelEx + programmatic serialization,
### gate + sigmoid before griddepcontrol.wait, same fast math): TIE with Triton, not worth a dispatch change
8 d per thread (128 threads per request): 14.43-14.66 in any CTA size. 4 d per thread (ndsc4, 256 threads per
request): CTA 32 14.35-14.45, CTA 64/128/256 13.92-14.29 = Triton _combine_hb (256 x 1 warp) 13.9. Check 8/8.
Ablation of the merge exposure (total over nds alone 12.1): PDL wait only +0.55 (without PDL +1.9); + m/l loads
+0.75; + the 1 MB of FP32 partials +1.75; + math ~+0.1. So the combine's cost is the post-wait L2 read of the partials
(~1 us for 1 MB: latency, not bandwidth) + PDL release ~0.5; nothing left to trim in the combine itself.

### nds at decode 16 re-priced (still loses to csl 9.4-9.7 H100)
S 8 1 CTA/SM: NS 4 kernel 8.62-8.94 (NS 3 was 9.8-10.7; NS 5 9.0-9.2) but + combine 10.82-11.20 total. S 16 NS 3
11.07-11.52. With the in-cluster merge (tools/a2/nds_cl_dead.cu): S 8 CL 8 NS 4 16.4-16.7, S 16 CL 16 15.3-15.5
(big-smem CTAs in 8/16-CTA clusters place badly). So for decode 16 an nds-like kernel would need a merge costing
< ~0.5 us exposed; neither a launch (~2.2) nor a last-CTA merge (~2) nor clusters does that.

### Takeaways / next
- Merging is the wall for split kernels at this size on the H100: ~2 us whether by PDL combine or in-kernel tail
  (PDL release ~0.5 + post-wait partial reads ~1.2). vtk (agent 1's verify-1 cluster kernel) at verify 16: 42.3 vs
  13.15 us (VTK_MAX_BATCH 16, 16 waves): dead.
  csl (one 16-CTA cluster of small CTAs per (request, KV head)) is the only design that hides it (decode 16).
- Next check (after 00:00Z): nds + lagS bundle as queued (decode 32 sliding nds, verify 16/32 LAG/nvs).
- Ideas left: (1) a csl-style small-CTA cluster kernel for decode 32 sliding was H200-tied (s4 csl32); (2) nds NS 4 is
  -1.2 us kernel at decode 16 vs NS 3: if a merge-free variant (one CTA per request, S 1, full window per CTA) could
  reach ~9 us... s7 says no-split gathers cap at ~8.3 us for B16 on H100: price nds at S 1 with NS 8 at decode 16
  (16 CTAs only: likely latency-bound, 32 tiles each).

## Session 12 (2026-10-25 11:56 - ~12:50 UTC): nds + nvs-verify-16 WIN on the H200 (check 9249146d); listed

Start: merged origin/main 069ee90 (agent 1 s109-s110) into agent2 (07d0de5); tests 16/16; H100 bench 2.038.
Account 18/20 checks; human 08:10Z + agent 1 s101 left the last 2 to agent2 -> used 1, left 1 for agent 1.

### Check 9249146d54844322 (agent2 07d0de5 = main 069ee90 + nds + nvs verify 16 + nvs LAG 2 at S 4): 2.527x
[2.489, 2.565], 1490.3 us, 16/16 (the account's best check; it also carries main's own unchecked changes).
| shape | H200 us | main's recent range | verdict |
|---|---|---|---|
| decode 32 sliding (nds) | **12.20** | 13.72-14.09 | WIN -1.7 (H100 said -1.17) |
| verify 16 sliding (nvs S 8) | **12.06** | 12.49-12.75 | WIN -0.4..-0.7 (baea7057 had nvs16 at 12.63 = tie: noise or LAG/stage changes since) |
| verify 32 sliding (nvs LAG 2) | 15.41 | 15.07-15.28 | tie/loss -> LAG 2 reverted (652f126) |
| decode 16 sliding (csl, unchanged) | 8.13 | 7.63-8.41 | - |
652f126 (LAG 1 at every S again; verify 32 = main's checked code): tests 16/16; listed in COORD "Ready to merge".
Lesson: nds (cp.async ring, 2 CTAs/SM, 256 CTAs) gained MORE on the H200 than on the H100, while csl32 (register
gathers, one-shot) lost its H100 win there: on the H200, ring designs with more bytes in flight per SM do relatively
better than the H100 predicts, by ~0.5 us at batch 32.

### Dead (H100, tools/a2/vab.py paired, 7 rounds)
- nds at decode 16 (CSL off): S 8 12.19, S 16 11.81, S 8 NS 4 12.19 vs csl 9.71 (+2.1..+2.5, 0/7). Even with a
  ~1.5 us H200 shift this loses.
- nds decode 32 NS 4 (smem -> 1 CTA/SM) +3.36; S 16 (512 CTAs) +3.12; NS 2 deadlocks (LAG 1 needs >= 3 slots;
  hung the GPU until killed).
- vab.py now renames any muse_nvs_* build name (was hard-coded to a2s9).

### Next
- Tomorrow's checks (agent2 <= 8): nothing queued yet. The batch 16/32 sliding shapes now sit at 8.1/12.2/12.1/15.4 us
  H200. Ideas with some value: (1) verify: CTA per (request, KV head, half window) with 2 consumer WGs merged in smem
  (halves partial bytes 4 -> 2 MB at verify 32; est. -0.5 us); (2) decode 16 csl is DRAM/latency-bound (all-L2 6.2 vs
  8.8 real, NOTES.md s75), no lever found.

## Session 13 (2026-10-25 12:27 - ~13:05 UTC): nvs NV_HALF = H100 WIN at verify 16 and 32 (H200 unchecked)

Start: origin/main 069ee90 already in agent2 (652f126 listed in COORD "Ready to merge"). Account 19/20 checks used
today (the last is agent 1's): **0 checks** this session.

### nvs NV_HALF (branch agent2-exp/half 0ae2b8a; submission.py `#define NV_HALF 1` in _NVS_SRC, NVS_HALF on the host)
CTA = (request, KV head H, split) instead of (request, split); consumer WG c takes half c of the split's keys for
head H (same 8 / 4 tiles per WG at verify 32 / 16 as before, same 128 CTAs), and WG 1 hands o / m / l to WG 0
through 18 KB of smem (bar.sync 4, 256) for the merge, so the FP32 partials and _combine_hb's S halve (verify 32
S 4 -> 2, verify 16 S 8 -> 4; 4 MB -> 2 MB). The producer's smem layout is unchanged: 64-B slot c of each staged
row holds key lo_c + 64 i + r of head H (half-row gathers: 2 x 64-B cp16 groups per key), so the consumer and the V
transpose are the 2-head code with "head" = half. Page-table ring stages both halves' 64 entries per tile; a WG with
fewer tiles than half 0 waits ready / arrives empty for the rest. New tokens: WG 1 of split S-1. 160 regs, no
spills (tools/a2/ptx.py: ptxas -v of _NVS_SRC). chk_nvs 7/7, tests 16/16, H100 bench 2.048 (s12 2.038).
| H100 paired vs 652f126 (tools/a2/vab.py, `name@file` variants added) | verify 16 sliding | verify 32 sliding |
|---|---|---|
| NV_HALF, NS 4 LAG 1 (12 rounds) | -0.54 (12/12) | +0.26..+0.35 (0/12) |
| NV_HALF + NS 5 LAG 2 (shipped on the branch, 9 rounds) | **-0.48 (9/9)** | **-0.27 (9/9)** |
On the half base at verify 32: NS 6 LAG 2 -0.62, NS 5 LAG 2 -0.77, NS 6 +0.11, NS 3 +0.19, NS 6 LAG 3 / NS 7 LAG 3
PFD 5 +0.05; at verify 16 all of these within noise except NS 7 (+0.5). After NS 5 LAG 2: PFD 3/6 noise, LAG 3 PFD 5
+0.82 (verify 32); _combine_hb HB 1 +0.13/+0.18, HB 4 +0.98/+0.40: HB 2 stays.
Caveat: LAG 2 on the OLD nvs was H100 -0.18 but H200 +0.2 at verify 32 (check 9249146d), so the H200 may differ.
Harness gotcha: `sed s/NV_HALF 1/NV_HALF 0/` also flips the host's `NVS_HALF = '#define NV_HALF 1' in _NVS_SRC`
string -> old kernel on the half grid (64 CTAs, +4.3 us): edit only the #define line.

### Next
- First agent2 check after 00:00Z: agent2-exp/half 0ae2b8a (verify 16 + verify 32 sliding). Keep per shape: if verify
  32 loses on the H200, try NV_HALF at verify 16 only (needs NV_HALF as a template parameter so both grids ship) or
  half with LAG 1.
- The same half split for nds (decode 32 sliding: 2 consumer WGs per CTA taking interleaved tiles, 1 CTA/SM, partials
  1 -> 0.5 MB, combine S 8 -> 4) is a bigger rewrite; expected <= ~0.3 us at 0.23 %/us. Not started.

## Session 14 (2026-10-25 13:02 - ~14:10 UTC): nvs cl (2-CTA cluster DSMEM merge at verify 32) = H100 WIN

Start: no new origin/main; account 19/20 checks (last one agent 1's): **0 checks**. Work on agent2-exp/half (merged
agent2 in, b3aa1dc), new commit **52b41bd**.

### nvs cl (NVS_CL_BATCH 32: NV_HALF at S 2 launched as 2-CTA clusters; no partials, no _combine_hb)
With NV_HALF, verify 32 has S 2: the 2 split-CTAs of each (request, KV head) form one cluster (cudaLaunchKernelEx +
cluster dim 2; 64 clusters fit, max 66). After the in-CTA half merge, rank r owns rows 32 r .. 32 r + 31 (warps 2 r,
2 r + 1); the other 2 warps stage their merged o / m / l in local smem (xo2), fence.proxy.async, bar 64, and one
thread issues a 9 KB `cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx` into the peer's xo2 (peer's
clbar: init 1 + its own arrive.expect_tx 9216 at kernel start; a barrier.cluster after init, ~0.1 us). Owners prefetch
their gates (raw bf16, before the half-merge bar), compute sigmoid while waiting, merge, o / l * v_scale, RMS-norm
(shfl over 4 lanes), * sigmoid(gate), BF16 store. chk_nvs 7/7, tests 16/16, bench 2.063 (s13 2.048).
| H100 paired, verify 32 sliding (cl vs NV_HALF + _combine_hb) | diff |
|---|---|
| final (bulk, symmetric) 3 runs | -0.86 (9/9), -0.29 (11/11), -0.56 (9/9) |
| verify 16 at S 2 cl (64 CTAs) | +1.86 (0/7): verify 16 keeps S 4 + combine |
| NS 4 LAG 1 / NS 5 LAG 1 / NS 6 LAG 2 on cl | +0.88 / +1.12 / -0.10 (5/7, noise; keep NS 5 LAG 2) |
How it got there (per-CTA %globaltimer stamps, tools/a2/clstamp.py with `#define NV_STAMP 1`; 32-ns resolution):
- first version (rank 1 pushes all 64 rows with st.shared::cluster + 128 per-thread arrive.release.cluster, rank 0
  does the whole epilogue) +1.22..+1.34 us. Cluster launch alone (combine kept): tie (16.29 vs 16.32).
- IEEE divisions (`vsv / l`, `y / (1 + exp)`) in the 128-thread epilogue cost ~1.4 us (latency-bound at 1 warp per
  scheduler): __fdividef (= Triton's div.full in _combine_hb) fixed it.
- the release-arrive push took ~1.6 us after rank 1's merge; one bulk copy cut it; the symmetric split halves the
  copy and the epilogue: -> win. Per-warp st.shared::cluster + 1 release arrive per warp: +0.46 vs bulk (dead).
- converting the gates to float before the half-merge bar.sync stalled the senders on DRAM latency (~0.8 us); load raw
  bf16 and convert after the push.
- remaining chain (cold, one run): rank merge -> peer wakes ~0.7-1.1 us (DSMEM delivery under load), epilogue
  ~0.4 us.
Caveat: NV_HALF itself is H200-unchecked (s13), and the H100 doesn't always predict the H200 for verify 32.

### Next
- First agent2 check after 00:00Z: agent2-exp/half **52b41bd** (verify 16 = NV_HALF S 4 + combine, verify 32 = cl).
  If verify 32 loses on the H200 vs 652f126's 15.07-15.41, fall back to NVS_CL_BATCH 9999 (NV_HALF + combine) or the
  old nvs (needs NV_HALF as a template parameter so verify 16 can keep it).
- The same cl exchange could replace _combine_hb wherever S == 2 fits one wave: nds (decode 32, S 8, 2 CTAs/SM) would
  need 8-CTA clusters (placement bad, s10/s12); a half/quarter nds with S 2 is a rewrite (consumers per CTA).

## Session 15 (2026-10-25 14:01 - ~14:35 UTC): NV_HALF listed; nvs cl split start barrier -0.2..-0.3 (H100); 2 merge ideas dead

Start: account checks 20/20 used today (0 left), so **0 checks**. Merged origin/main d93a80d (agent 1 s111-s113) into
agent2 (3aaa979) and into agent2-exp/half (92118bc). Agent 1's check **2d0bfdc6** (main + 0ae2b8a's NV_HALF, s112) gave
H200 verify 16 sliding **11.63** (e64d3450 11.90: WIN) and verify 32 15.43 (15.42: tie). Listed in COORD "Ready to
merge": **agent2-exp/nvhalf 7ba35f9** = origin/main d93a80d + exactly 0ae2b8a's submission.py (tests 16/16); agent2
now = 7ba35f9.

### WIN (H100, unchecked): nvs cl start barrier split (agent2-exp/half e5f65ed)
cl's kernel-start `barrier.cluster.arrive` + `wait` (makes the peer's clbar init visible before the DSMEM bulk copy)
-> arrive.release at start, `barrier.cluster.wait.acquire` only in the 2 sending warps right before their copy
(csl already does this). Paired vs 52b41bd-equivalent: verify 32 sliding **-0.29 (7/9), -0.19 (10/11)**; chk_nvs 7/7,
tests 16/16. Found via the nds pair probe below (nds's start cluster barrier alone cost ~0.45 us).

### DEAD (H100 paired, tools/a2/vab.py)
- **cl 2 at verify 16** (NV_HALF S 4 as 2-CTA clusters = split pairs, cl exchange in-pair, then a per-warp global
  ticket: each owner warp __stcg's its pair-merged rows, fence + atom.add.acq_rel.gpu; the later pair's warp merges
  the other's rows from L2 and finishes; no _combine_hb; tools/a2/nvs_cl4t_dead.patch): **+2.59 (0/9)** vs S 4 +
  combine. Ablation: cluster launch only (combine kept) tie (+0.03); no fences +1.8; no wait at all (pair 1 always
  finishes, racy) still +2.5: the store -> fence -> atomic -> fence -> load chain (~3 serialized L2 trips) costs more
  than the PDL combine it replaces. Same lesson as agent 1's s28 sfuse / s34 flag merges: no global cross-CTA sync.
- **nds pair merge** (decode 32 sliding: split pairs as 2-CTA clusters, symmetric: rank r keeps KV head r, one 2.5 KB
  bulk copy sends the other head's rows to the peer, partials + _combine_hb at S 4 instead of 8;
  tools/a2/nds_pair_dead.patch, correct on 5 decode cases): **+1.06 (0/9), +1.09 (0/7)**; without the start
  barrier.cluster (racy) still +0.45. Cluster-2 launch of nds alone (cudaLaunchKernelEx): +0.08 (noise). Halving the
  combine's partials saves less than the DSMEM round trip costs; with a split barrier it would still be ~+0.5.

### Next
- First agent2 check after 00:00Z 2026-10-26: **agent2-exp/half e5f65ed** (verify 16 = NV_HALF S 4 + combine, already
  H200-won; verify 32 = cl + split barrier). If verify 32 wins vs 15.4, list it; else list only NV_HALF (already listed).
- Merge structure summary for batched sliding (H100): PDL combine ~2 us; in-cluster DSMEM exchange ~1 us chain (only
  wins when it removes the combine launch entirely: cl at S 2); global ticket/flag chains ~2.5 us. S 2 cl fits only
  verify 32 (128 CTAs, 1/SM); 4-CTA clusters cap at 30 (< 32 needed at verify 16).

## Session 16 (2026-10-25 14:31 - ~15:10 UTC): nds split pairs in one CTA (smem merge) DEAD

Start: account 20/20 checks used today, sandbox deleted ~17:45Z (COORD human 14:10Z): **0 checks**, H100 only. Merged
origin/main c6f8370 into agent2 (345e974; submission.py unchanged vs 7ba35f9, tests 16/16) and agent2 into
agent2-exp/half (d7dd5a4; nvs conflicts resolved to half's cl superset = e5f65ed's submission.py).

### DEAD: nds NV_PAIR (agent2-exp/ndp, tools/a2/nds_pair_cta_dead.patch, stamps tools/a2/ndpstamp.py)
Decode 32 sliding: the 2 CTAs that already share an SM (nds: 256 CTAs, 2/SM) fused into one 512-thread CTA running
2 adjacent splits (2 producer + 2 consumer warpgroups, 2 Smem copies, own named barriers); the odd split's live warp
hands o / m / l to the even one through smem (bar.arrive / bar.sync 5, 64), which merges and writes partial slot
split / 2: partials 1 -> 0.5 MB, _combine_hb S 8 -> 4. Correct (chk.py 12/12).
| H100 paired (tools/a2/vab.py), decode 32 sliding | diff vs nds |
|---|---|
| NV_PAIR | **+0.74 (0/9), +0.72 (0/7)** |
| pair launch, no merge (S 8 partials) | +0.96 (0/7): the packaging alone |
So the in-CTA merge is worth ~-0.3 (combine 1.55 -> 1.58 us warm: still latency-bound, the gain is the 8 -> 4 slot
tail), but one 16-warp CTA runs each split's key loop ~0.5 us slower than two 8-warp CTAs (stamps: first tile at
2.05 us in both, loop median 3.33 -> 3.84 us; profiler warm kernel 7.71 -> 8.65). NOT explained by: the live softmax
warps sharing a sub-partition (moved odd split's rows to warp 1: same), register-held barrier ids (per-pv copies with
immediates: same), request locality (odd vCTA on a far split: same), smem carveout (205 vs 202 KB, both 228), regs /
spills (96, LOCAL 0 both). Lesson: keep nds-style producer/consumer splits as separate CTAs.

### Next (unchanged from s15)
- First agent2 check after 00:00Z 2026-10-26: **agent2-exp/half e5f65ed** (verify 32 sliding = cl + split barrier;
  verify 16 = NV_HALF, already merged to main via 7ba35f9). Compare verify 32 sliding vs ~15.4 H200.
- Batched-sliding merge options are now all priced on the H100 (PDL combine ~2, DSMEM ~1 chain, global ticket ~2.5,
  same-SM packaging +1). Only cl at S 2 (verify 32) won.

## Session 17 (2026-10-25 14:55 - ~15:30 UTC): page-table L2 prefetch (decode 16/32 small win, verify loses)

Start: account 20/20 checks used today, sandbox deleted ~17:45Z: **0 checks**, H100 only. Merged origin/main 00a42b5
(agent 1 s118) into agent2 (12911f3; submission.py batched-sliding code unchanged).

### Idea: L2-prefetch the plausible page-table span while seq_lens is in flight
Every batched sliding kernel runs seq_lens -> page table -> K/V rows: 3 dependent cold trips. With agentic lengths
(reference.py: seq in ctx +/- 7.5 %, table width = longest + L) the window start is >= 0.86 x width - 2112, so at
kernel start each CTA issues one `cp.async.bulk.prefetch.L2.global` over its 1/rq share of [0.86 maxp - 2112, maxp)
of its request's page-table row (~56 KB per request: 0.9 / 1.8 MB at batch 16 / 32). The dependent page-table read
then hits L2. A hint only: results bit-identical, decisions on shapes only; useless (not wrong) on other spreads.
| H100 paired (tools/a2/vab.py) | diff |
|---|---|
| csl (decode 16 sliding), thread 32 | **-0.13 (10/11), -0.11 (10/11)** |
| nds (decode 32 sliding), thread 32 | -0.24 (9/9), -0.11 (8/11), -0.06 (6/11) |
| nds, issued by consumer warp 1 (thread 160) | ~tie vs thread 32 |
| nvs (verify 16 / 32), thread 32 | **+0.18 (0/9) / +0.19 (1/9)** |
| nvs, thread 160 | +0.32 / +0.35 (0/9) |
Verify loses (probably the extra DRAM bytes against a gather-bandwidth-bound nvs; decode kernels are latency-bound).
Kept on csl (spec < 0 only) and nds. Branch **agent2-exp/pf 946a445** (= agent2 now), tests 16/16.
Bundle for the check: **agent2-exp/halfpf 6a9a26a** = agent2-exp/half e5f65ed (verify 32 cl) + pf; tests 16/16,
H100 bench decode 16/32 sliding 9.52/13.42, verify 16/32 sliding 13.14/15.87.

### Next
- First agent2 check after 00:00Z 2026-10-26: **agent2-exp/halfpf 6a9a26a**. Per shape vs 652f126/nvhalf on the
  H200: verify 32 sliding (cl) vs ~15.4; decode 16 sliding (pf) vs 8.13 (noisy, 7.63-8.41 across runs); decode 32
  sliding (pf) vs 12.20. List only the shapes that win (pf can be dropped per kernel: delete the A2 prefetch block).
- Same trick elsewhere: agent 1's batched full layers (_gd_full / _gw_verify) read seq -> page table too, but
  their page-table reads are long streams (not on a single chain); probably nothing there.

## Session 18 (2026-10-25 15:12 - ~15:40 UTC): decode 32 head-chain / CTA-order probes, all dead (0 checks)

Start: account 20/20 checks used today, sandbox deleted ~17:45Z: **0 checks**, H100 only. Merged origin/main 28ba08b
(agent 1 s119/s120 tools) into agent2; submission.py unchanged.

H100 paired (tools/a2/vab.py, 9 rounds), decode 32 sliding (nds), timing-only probes:
| variant | diff vs nds |
|---|---|
| constant RoPE position (removes the seq -> cos/sin trip; also makes that row L2-hot) | -0.13 (7/9): upper bound of a cos/sin L2 prefetch (~3 MB span at ctx 80K): not worth bytes or a check |
| split-major CTA order (request = blockIdx % B) | **+0.35 (1/9)** |
| even splits in grid half 0, odd in half 1 (pair partners = same request) | +0.11 (1/9) |
So the request-major order (blockIdx / S = request) is best and the consumer's cos/sin trip is not on the
critical path (the producer chain seq -> page table -> rows is). No change shipped.

### Next (unchanged)
- First agent2 check after 00:00Z 2026-10-26: **agent2-exp/halfpf 6a9a26a** (verify 32 cl; decode 16/32 pf).

## Session 19 (2026-10-25 15:33 - ~16:00 UTC): nds K-early barrier = small H100 win at decode 32 (0 checks)

Start: account 20/20 checks used today, sandbox deleted ~17:45Z: **0 checks**, H100 only. Merged origin/main 4c00ee9
(agent 1 s121/s122 tools/notes) into agent2 (0a8d56c); submission.py unchanged by it.

### K-early barrier in nds (decode 32 sliding), agent2 565c536
The producer arrives `ready[s]` right after its `cp.async.wait_group` (+ fence.proxy.async), BEFORE the bar.sync and
the V transpose, then arrives a new `readyv[s]` after the transpose. The consumer waits `ready` before QK(i) (as
before) and `readyv[sp]` between issuing QK(i) and PV(i-1) (and in the last-tile PV): the V transpose leaves the
gather -> QK chain. No barrier between softmax and P@V (agent 1's s46 lesson in NOTES.md: that costs more there).
Results bit-identical in kind (same math), tests 16/16, chk.py 12/12.
| H100 paired (tools/a2/vab.py) | diff |
|---|---|
| nds kr vs agent2 (decode 32), 11 rounds x2 | **-0.06 (10/11), -0.10 (10/11)** |
| nds kr vs halfpf 6a9a26a (decode 32), 15 rounds | **-0.14 (13/15)** |
| same in nvs (verify 16 / 32, NV_HALF base) | +0.10 (3/11) / +0.19 (2/11): not applied (nvs is gather-throughput-bound) |
| nds kr + K and V in separate cp.async groups (wait K first) | +0.24 (1/11): dead |
Gotcha: adding `readyv` to nvs's Smem moved `xo` off 16-B alignment ("misaligned address" from its float4 stores):
any Smem field added before xo needs `alignas(16) float xo[...]`.
Check candidate: **agent2-exp/halfpfkr abc5f38** = halfpf 6a9a26a + kr (merge of agent2 565c536); tests 16/16.

### Next
- First agent2 check after 00:00Z 2026-10-26: **agent2-exp/halfpfkr abc5f38** (replaces halfpf 6a9a26a: same code +
  nds kr). Per shape vs the H200: verify 32 sliding (cl) vs ~15.4, decode 16 sliding (pf) vs 8.13, decode 32 sliding
  (pf + kr) vs 12.20. List only the winners.

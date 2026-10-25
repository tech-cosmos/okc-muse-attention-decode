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

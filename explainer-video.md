# Explainer video (Cutscene)

A 2:50 narrated explainer of GPU kernel engineering for web developers, made with cutscene.dev
(studio speed, video_type explainer). It covers kernels/threads/SMs, the memory hierarchy, what attention
decode computes, split-and-merge and launch overhead, the toolbox (Triton, Gluon, CUDA, PTX, wgmma,
clusters/DSMEM, warp specialization, CUDA graphs), and scoring plus the measure loop. No repo code or files were
sent; only a description of the concepts.

## Versions

| Version | Job | Page | Local file | Status |
|---|---|---|---|---|
| v1 | `91bd669f1042` | https://cutscene.dev/v/91bd669f1042 | `~/Downloads/gpu-kernels-explainer-cutscene.mp4` | Superseded: wrong memory floor |
| v2 | `2f4acf79ef3c` | https://cutscene.dev/v/2f4acf79ef3c | `~/Downloads/gpu-kernels-explainer-cutscene-v2.mp4` | Superseded: wires under SM squares |
| v3 | `ead8092c62b5` | https://cutscene.dev/v/ead8092c62b5 | `~/Downloads/gpu-kernels-explainer-cutscene-v3.mp4` | **Current. Verified, OK to post** |

Download links from `get_status` are valid 7 days; the pages are permanent (private until sharing is turned on).
The Remotion source project is not accessible: Cutscene's tools and docs only deliver the MP4, the page and
a report. Changes go through `revise_video` notes on the latest job.

## Fixes

v1 -> v2 (all three confirmed in v2 stills; audio byte-identical to v1; frames differ only at ~1:48-1:56,
~2:25-2:28 and ~2:45-2:50):

1. Memory floor in the launch-overhead timeline (~1:48-1:56): v1 said `≈ 8.5 µs`, `41 MB ÷ 4.8 TB/s (head size
   128)`. This problem uses head size 64 (`HEAD_DIM = 64` in reference.py), so the batch-1 full layer moves ~20.5 MB
   and the floor is ≈ 4.3 µs (TASK.md's table agrees). Now `memory floor ≈ 4.3 µs`, `≈20 MB ÷ 4.8 TB/s (batch-1
   full layer, head size 64)`, with the dashed line at 4.3 µs.
2. End card: removed the made-up grid logo beside "Open Kernel Challenge" (it could pass as official branding).
3. Score line: `geomean(speedup vs SGLang)` -> `geomean of speedup over 6 weighted loads` (each load = 46 x sliding
   + 16 x full). The narration still says "geometric mean speedup over SGLang", which is fine at that level.

v2 -> v3 (verified 2026-10-27):

4. Wires drawn under the SM squares (S07, chapter 04 "Why it's hard", ~1:30-1:35, the "1 request -> a few blocks /
   most SMs idle" beat). The two orange wires from the context bar to the two lit green SMs in row 2 pass
   *behind* the empty row-1 squares, so the line and packet dots look dimmed or hidden. Asked for the wires and
   packets to be drawn on top of the chip and squares for their whole path, with the same paths, colors and
   timing. Also asked Cutscene to fix the cyan split-context packets in the next beat if they have the same
   layering problem. Spotted by the user, not caught in the v1 review (frame sampling every 3 s, contact sheets
   at half size: too coarse to notice z-order).

## v3 check (2026-10-27)

- Fix confirmed: full-size crops of v2 vs v3 at 1:32.3 and 1:33.5 show the orange wires and packet dots dimmed under
  the row-1 squares in v2, and bright on top of them in v3, landing on the two green SMs. Cutscene's report: the
  fix was a z-order change only (the wire block is now drawn after the SM tiles in the S07 scene).
- Cyan split-context packets: Cutscene checked them and they were already drawn above the tiles, so no change.
- Audio track md5 identical to v2 (and v1): `76b5128731997e4121ba40cd276cea33`.
- Frames vs v2 (PSNR at 10 fps): the only real change is at 1:29.7-1:33.5 (PSNR 47-52, max pixel diff 91/255).
  A few other stretches (0:00-0:08, 0:59-1:05, 1:34-1:39) read PSNR 53-58 with max diff ~35/255; a 6x-amplified
  difference image shows faint even outlines on every edge in the frame, which is re-encode noise, not content.
  So the v2 fixes (4.3 µs floor, no logo, score line) carry over unchanged.
- Lesson for future checks: sample suspect beats at full resolution and 10 fps, and look at crops where lines
  cross other shapes. Contact sheets at 1 frame per 3 s missed this layering bug.

## Known simplifications left in (acceptable for an intro)

- "Decode is memory bound": verify-full is compute-bound on the H200.
- The hook's "8 layers" and "≈10 µs per layer" are illustrative.
- Whisper transcript quirks ("L2Cacher", "SG-LANG", "hide every weight" = wait) are transcription errors, not
  narration errors.

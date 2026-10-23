Codex session 5: native reduction follow-up
=========================================

No new production kernel is promoted by these experiments. Start from codex
submission.py blob a66802981a86065ee7647e4b773b9d811db7ac567a76983d544a42fc7a6a59b1
(SHA256, effc2bf/5e2441f source). Triton attention and plans are unchanged.

Run from repository root with /home/daytona/okc activated:
  python experiments/reduce/session5/paired.py
  python experiments/reduce/session5/pregate_paired.py <new-json-output-path>
  python experiments/reduce/session5/main_paired.py
  python experiments/reduce/session5/validate_variant.py cooperative full
  python experiments/reduce/session5/validate_variant.py cooperative fresh

The session4 dependencies in ../ were merged from codex-exp/cuda-reduce.
The .cu snapshots retain the generated experimental native sources. These are
research scripts only; submission.py does not import any experiments.

Method
------
All GPU setup, correctness work, and timing take /tmp/okc-gpu.lock. Pre-captured
kernel_fn graphs replay in rotated/reversed order, with 512 MB L2 flushes before
every replay, 100 rounds, and interleaved empty graph measurements. Results include
raw times, separate medians minus empty replay, and median within-round differences.
The latter need not equal the difference of two separate medians. Every compared
case passes the independent reference, not just a baseline comparison. Cases cover
B1/B2, L1/L8, page1/page64, contexts 1500/40K/80K/120K. No plans are overridden.

initial-compare.* reruns the old five-pair timing method. Its result was copied
out of ../results/cooperative.json and the historical file restored unchanged.
paired-harness-incomplete.log records a DRIVER error: captured graphs had not
been replayed before examining their output buffers. The resulting NaNs were
uninitialized output, not a native kernel failure. The corrected paired.py
explicitly replays every graph before correctness checking. No timings from the
incomplete attempt are used.

Findings
--------
- Cooperative warp-zero weights do not reproduce the earlier apparent 0.4 us
  decode win. Initial long-context medians were 17.216 -> 17.296 us decode B1,
  18.304 -> 18.432 verify B1, 38.816 -> 38.992 verify B2. The tightly paired run
  was +0.144/+0.240/+0.208 us relative to production on those cases. Retain the
  existing scalar warp-zero epilogue.
- Address/stride arithmetic in 64-bit pointer space, and a warp-uniform branch
  removing masks for complete 16-split groups, generally lose ~0.25-0.46 us on
  small scored grids. The isolated B2 verify unmasked improvement is not enough
  to justify a new dispatch. These variants preserve the exact same FP32 math.
- Computing sigmoid(gate) in registers before the PDL wait does NOT force actual
  overlap: ptxas moves its EX2 instructions after ACQBULK, even with an empty
  volatile asm consuming the results. Pregate SASS confirms zero EX2 before the
  wait, eleven after. Do not call this a successful early-computation pipeline.
- Publishing the precomputed sigmoid through volatile shared memory DOES force
  it ahead of ACQBULK (two EX2 and two STS before wait, nine EX2 after; no coherent
  partial reads before wait). This version saves only ~0.096 us on B1 decode,
  ties/slightly loses B1 verify (+0.048 us), and is inconsistent across nearby
  shapes. No convincing scored-load gain; keep production.
- Existing production still beats the inherited main full merge: main-to-native
  medians 16.928 -> 16.272 us B1 decode, 18.464 -> 17.808 B1 verify,
  39.184 -> 38.496 B2 verify; paired differences 0.656/0.672/0.672 us.
  Nearby comparisons also favor the current native code by ~0.37-0.64 us.

Validation scope
----------------
cooperative-full.log: 16/16 independent reference tests.
cooperative-fresh.log: 72 fresh data/routing/length graph replays across nine
native/fallback cases, all pass (including empty shortened contexts, pages1/64,
scales1 and1/64). These are end-to-end dispatch tests, not 72 native-only calls.
Other experimental variants: six independent reference comparisons each plus
100-round repeated graph comparisons. No full-suite, fresh-data, or sanitizer
coverage is claimed for address/unmasked/pregate variants. None were submitted.
The unchanged production native code retains session4's validation/sanitizers.

Judge jobs, final source state, and day budget are recorded in NOTES-codex.md.
Do not resubmit pending checks. This directory's status JSONs are snapshots.

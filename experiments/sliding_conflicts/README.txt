Session 23: native sliding copy-counter attribution, 2026-10-24

Outcome: no candidate and no production change. The previously reported 4104
"L1 Wavefronts Shared Excessive" do not establish 4104 removable shared-bank
conflicts. They are all attributed to the four LDGSTS instructions. The actual
native kernel's ordinary LDS, STS and WGMMA instructions have zero source excess.

Reproducing the native profile gives 4104 copy source excess / 4112 ideal, with
zero hardware bank-conflict counts for LDGSTS, LDS, STS and GMMA. A zero LDGSTS
hardware count alone is NOT sufficient evidence: it also stays zero in the
deliberately bad-layout control below.

The diagnostic copy.cu varies source and destination separately. Each 128-thread
CTA copies 64 x 64 bytes using two cp.async.cg 16-byte instructions/thread; 64
CTAs. Sources: contiguous 64-byte rows, first 64 bytes of successive 128-byte
rows, or first 64 bytes of permuted 128-byte rows. Destinations: linear, the
production K SW64 mapping, or a deliberately conflicting 128-byte spacing for
each 16-byte piece. All output bytes are restored to logical order and compared
with an independent CPU result.

LDGSTS source-counter excess, summed over the two instruction sites:

                           linear     SW64    bad-spacing control
contiguous global rows          0        0                 14336
128-byte global row stride   2048     2048                 14336
permuted 128-byte rows        2048     2048                 14336

All entries have 2048 ideal copy wavefronts. The bad-spacing control additionally
has 8192 excess LDS wavefronts and 8192 LDS hardware conflicts. All LDGSTS hardware
conflict counters remain zero. Thus source geometry alone changes the derived
copy metric with the shared mapping fixed. This is direct evidence against
interpreting the native copy number as purely shared-destination serialization.
It does not prove a precise microarchitectural implementation, an isolated copy
latency, or a useful end-to-end speedup from changing global gather topology.

No layout variant of kernel_fn was created. No new full-suite, production bench,
or kernel_fn graph-timing claim: production is byte-identical to the validated
session21 source, SHA256 6b171beb8ed5e2a306ff67c3820ba7e83943f73c1a73d7b06eeeb630e0d0eecf.
The copy probes are offline diagnostics, not complete attention implementations
and never eligible for submission. No input-dependent work was moved out of any
timed production call. No Triton/Gluon code or plans were tuned.

Validation: 27 fresh-data graph replays, 3 per probe, each checks all 262144 output
bytes. The final permutation/SW64 probe additionally passes memcheck, synccheck
and racecheck; each tool executes those 3 fresh graph replays and one ordinary
profile call. This is diagnostic-only sanitizer coverage, not new native-attention
sanitizer coverage. Both diagnostic drivers hold /tmp/okc-gpu.lock across GPU
allocation, generation, capture, validation and profiling. No profiler duration
is used as a latency measurement.

Reproduce from repo root:
  sh experiments/sliding_conflicts/run.sh
  python experiments/sliding_conflicts/audit.py

run.sh builds the standalone probe and profiles all nine controls, then profiles
the exact native CUDA extracted from submission.py using the isolated extension
build directory. audit.py verifies source identity, completed checks and counter
relationships. The raw ncu-rep binaries and executable are ignored; readable
source/raw CSVs, logs and source are archived. The original session21 profile is
also retained for attribution comparison. Final reports come from the complete
run.sh execution. No speedup or occupancy claim.

Relevant NVIDIA documentation (consulted 2026-10-24):
https://docs.nvidia.com/nsight-compute/NsightCompute/
  Documents the source-view derived metrics separately from hardware counters.
https://docs.nvidia.com/nsight-compute/ComputeTriage/
  Discusses source-sector patterns, LDGSTS and shared/L1 client arbitration.

Next: do not reopen shared-layout sweeps based on the 4104 number alone. A future
native gathered-attention architecture needs complete graph measurements against
production, preserving global source routing and all attention work. The earlier
paired-head/full-row native designs remain rejected on their recorded timings.

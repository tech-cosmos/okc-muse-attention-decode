Session15: rejected native sliding L1 last-CTA completion/reduction

Production base: 8a93715, submission source 3fb391f, SHA256
16a7db558d53ddc110e2bf6ccec1ff243f4407b817ec939eeee409fbe2b8f16e.
No production changes or new judge requests. Only this experiment branch includes
these files. Current Triton kernels and plans are unchanged.

Implementation
--------------
generate.py extracts the embedded native single-tile L1 kernel, preserving its
attention math. After writing FP32 partials, writers threadfence, synchronize,
and one leader publishes via a device-scoped acquire/release atomic. The last
CTA for a request/KV head reduces all splits, RMS normalizes, gates and writes
BF16 output. It resets the scheduling counter before the kernel completes.
Counter storage caches only scheduling state (reset to zero each invocation),
never anything derived from input tensors. No CTA spins or waits for another.
Partial reads are volatile coherent inline-PTX loads after completion detection.

native.py: immediate load/weight/reduce per split.
prefetch.py -> prefetched.py: separate source loops for partial loads and
reduction. SASS issues nine vector partial loads before the first reduction
FFMA (one in the original), then interleaves further loads and math. It does not
hold every partial in registers at once. 74 -> 78 registers, no spills.
The separate-merge control retains the original PDL merge and attention math.
Both experimental paths dispatch to all L1 sliding group8/D64/page1 or64 cases;
this broad dispatch is for measurement/validation only and is not promoted.

diagnostic.py builds three offline modes: attention only; attention with
completion tracking/counter reset but no reduction; complete fused attention.
The first two deliberately omit the final output and CANNOT be submitted as
kernel_fn. It independently reduces partials with torch and checks reference
outputs/cache before and after graph capture and after timing. This is only a
cost decomposition, never a scored candidate or a performance claim for a
correct complete implementation.

Reproduce (activate ~/okc)
-------------------------
python experiments/sliding_completion/generate.py
python experiments/sliding_completion/prefetch.py
python experiments/sliding_completion/validate.py
python experiments/sliding_completion/validate_prefetched.py
python experiments/sliding_completion/compare.py
python experiments/sliding_completion/compare_prefetched.py
python experiments/sliding_completion/fresh.py
python experiments/sliding_completion/fresh_prefetched.py
python experiments/sliding_completion/sanitize.py
python experiments/sliding_completion/sanitize_prefetched.py
python experiments/sliding_completion/diagnostic.py
python experiments/sliding_completion/bench.py
python experiments/sliding_completion/inspect_sass.py
python experiments/sliding_completion/audit.py

Run sanitize scripts under compute-sanitizer --tool memcheck|synccheck|racecheck
--error-exitcode 1. Native-only calls avoid the unrelated Triton loader probe.
Use cuobjdump -sass and -res-usage on muse_completion_v1 and
muse_completion_prefetched_v1 for the archived disassembly/resources.

All custom GPU drivers hold /tmp/okc-gpu.lock through input generation, reference
checks, capture and replay. Comparisons flush512MB before every replay and
subtract an empty replay. They use40 balanced adjacent AB/BA pairs per entry.
Diagnostic uses60 pairs. Full bench retains tools/bench.py timing-lock behavior
and was started after comparisons, validation and sanitizers completed. No NCU
profiling was needed; SASS/resource inspection does not run kernels.

Results (H100 microseconds, each candidate compared with its own paired parent)
----------------------------------------------------------------------------
                            parent/original-fused    parent/prefetched-fused
B1 ctx80000 page1              4.384 /10.432             4.416 /6.576
B2 ctx80000 page1              4.992 /10.944             5.040 /6.848
B16 ctx80000 page1            10.416 /18.512            10.336 /14.992
B32 ctx80000 page1            15.712 /27.264            15.792 /23.936
All nearby comparisons also lose, including B1 ctx10 and page64 cases.
Raw pairs/deltas are in results/comparison*.json. Median deltas need not equal
the differences of individual medians.

Diagnostic B1: attention3.232, completion-only4.608, full fused6.624us.
B2: attention3.488/completion-only4.864; attention3.424/full fused6.944.
Completion tracking adds paired1.376/1.408us before any final reduction.
This is the combined cost of fences, synchronization, atomic publication and
last-CTA detection/reset; it does not isolate the atomic instruction's latency.
Thus the redesigned reduction still cannot make this completion scheme win.

Validation: each candidate16/16 tests (3 native,13 fallbacks),48 fresh graph
replays (8 all-native cases, six seeds incl empty contexts), eight reference
comparisons before/after capture plus eight controls. Each sanitizer/candidate
checks24 native calls (two contexts, two pages, two scales, three fresh QKV
calls); memcheck/synccheck0 errors, racecheck0 hazards/errors/warnings.
Counters are checked zero after fresh and sanitizer calls. Tests cover TP8/DP,
pages1/64, scales1/1/64, soft/coherent/edge inputs, changed QKV/cache/routing/
lengths/positions/cache slots. Full benches and source audits are archived too.

Control observation: at B3 ctx4000 page1, the existing native separate-merge
kernel (outside production's B*Hkv<=4 dispatch) is ~5.2us vs ~6.2us parent in
both comparisons. This is an unqualified nearby-shape lead, not a scored gain;
no dispatch extension or sweep was made. B16/32 control still loses.
Close last-CTA fusion on this L1 attention design. Further stage/register/merge
sweeps are not justified by these results.

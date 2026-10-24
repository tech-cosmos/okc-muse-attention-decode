Session 22: full-verify merge critical path and native grouped-row reduction

Production submission.py is unchanged SHA256
6b171beb8ed5e2a306ff67c3820ba7e83943f73c1a73d7b06eeeb630e0d0eecf.
No Triton/Gluon attention, plan, or merge source is changed. These files are
experiments and offline diagnostics; none is imported by production.

Diagnostic.py intercepts the merge launch only, comparing complete production
with merge omission and with a serialized merge. Omission is NOT a valid
submission: it returns no final output. Its fresh FP32 partials are independently
reduced in PyTorch and checked against the attention/cache reference outside
timing. Serial still computes the full valid result. Each graph performs all
of its input-dependent work on every replay. No cached result is submitted.

native.cu: four independent output rows per 128-thread CTA, ordered physically
by partial layout. The warp layout owns one row/warp, two dimensions/lane,
loads FP32 partial vectors before weighted summation, and shuffles split weights.
The striped layout divides each warp into four split groups x eight dimension
groups, with eight dimensions/lane and vectorized 128-bit loads. Two shuffle
reductions combine split groups. Both retain FP32 max/sum/accumulators, the same
stable softmax merge, RMSNorm/gating and BF16 output. No keys/values are skipped.
Coherent volatile inline PTX loads follow griddepcontrol.wait in SASS. Native
launches preserve the Gluon chain's PDL dependency; there is no new semaphore.
CAP is a shape-derived 16/32/64 capacity, with runtime masking for actual S.

candidate.py replaces only the final merge of existing Gluon-dispatched calls.
It leaves fallback shapes alone. Standard 16-case tests do not enter this path;
validate.py adds 42 actual native fresh-input graph replays per variant, changing
all tensors/routing/lengths/cache slots, including empty contexts. Synthetic.py
isolates native reduction from Triton for sanitizer runs and independent Torch
reference checks on 48 replays per tool, including capacities not used by the
production Gluon dispatch. It does not test attention/cache writes itself.

Reproduce from repository root with ~/okc/bin/python and
TORCH_EXTENSIONS_DIR=$PWD/.torch_extensions_session22:
  experiments/verify_merge/diagnostic.py
  experiments/verify_merge/pairs.py --variant warp
  experiments/verify_merge/run_cycle.py
All custom GPU drivers lock /tmp/okc-gpu.lock during input generation, reference,
capture and replay; comparisons flush 512 MB before each replay, record balanced
adjacent AB/BA pairs, and subtract an empty replay. Full benches keep bench.py's
own timing-loop lock and are run sequentially with our other GPU work.
No NCU duration is used as graph timing. Native module builds at import only.

Source compiler resource reports and full SASS are in results/. sass_audit.py
checks partial-load ordering for all six compiled native specializations.
The result table and final validation/budget summary are in NOTES-codex.md.

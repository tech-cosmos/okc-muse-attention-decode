Session 6: separate whole-cache FP8 packing (rejected; no production changes)

A bit-exact native gather materializes every historical K/V row in logical order
into contiguous FP8 scratch. New-token insertion and attention are not included:
this deliberately measures a necessary prefix of a proposed gather-then-dense
attention implementation. No attention speedup is claimed.

Run from repository root with ~/okc activated:
  python experiments/pack/bench_pack.py
  python experiments/pack/validate_pack.py

All times: H100 CUDA graph replay, cold 512 MiB L2 flush, median, empty replay
subtracted, tools/bench.py GPU lock. 36 copy configurations per case (vector bytes
4/8/16, unroll 1/2/4/8, total CTAs ~264/528/1056); 108 bit-exact comparisons.
The best observed copy-only times versus complete production verify attention:
  batch  1: 16.288 us / 18.272 us (20,480,000 source bytes)
  batch 16: 246.816 us / 154.848 us (330,701,056 source bytes)
  batch 32: 494.144 us / 305.488 us (662,059,008 source bytes)
These are sweep minima, not paired aggregate speedups. The large-batch copy alone
is already ~60% slower than the entire existing call. Dense attention was not
implemented or installed; the prefix cost rules out this materialization design.

Fresh validation: 288 bit-exact graph checks across 8 shapes (pages 1/64, TP8/DP,
batches 1/2/3/4, contexts 1/257/2000/6000), all 12 vector/unroll variants. Change
cache bytes and page routes, include empty contexts, verify unused output suffix
retains its sentinel. This is a cache-copy test, not an attention reference suite.
No sanitizer coverage is claimed for packing. Never dispatched or submitted.

The script uses 32-bit unsigned offsets for this experimental size range; it is
not a generic large-allocation library. No production imports depend on it.

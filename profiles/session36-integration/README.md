# Session 36 main integration

Merged main32b1762/da388d1 into Codex. Main stores unnormalized split
accumulators in BF16 on selected Triton paths, with FP32 arithmetic and
max/sum storage. Its Gluon and small sliding paths retain FP32 storage.

The native full merger requires `data_ptr<float>()`. The integration preserves
FP32 producer storage under its existing shape predicate, factored into
`native_full`; native CUDA and dispatch coverage do not change. Other paths
inherit main's dtype selection. All20 shared Python helpers match main exactly.
`audit.py` also checks both native extension ASTs against previous production.

Exact tested source is `submission.py` here, SHA256
`793a8476e50a4716b562abae1f1ce0609343d38cb444bedcc0303739047baa70`.
`previous.py` is9dbf51a/SHA62fefe, and `main.py` is main da388d1. The inherited
update is reproducible with `git diff a8de0e7..da388d1 -- submission.py`.

Validation:16/16 and all12 benchmark shapes, local aggregate1.987x against
fixed SGLang/H200 times. There are128fresh graph replays (64per source),
2080adjacent AB/BA pairs and156independent output/cache comparisons. These
counts combine16previous/current cases and10main/current cases. Fresh tests
change QKV/cache/routes/lengths/positions/cache slots, including empty contexts.
All custom drivers lock generation/reference/capture/replay and flush512MB
before each timed graph; empty replay is subtracted. Standard benchmark uses
its timing-loop lock. Native kernels unchanged: no new integrated sanitizer claim.

Paired candidate-minus-previous deltas, us:

| Case | Previous | Integrated | Paired delta |
|---|---:|---:|---:|
| decode16full |130.432|129.360|-0.912|
| decode32full |242.816|241.392|-1.696|
| verify16sliding |12.512|12.384|-0.176|
| verify32sliding |20.192|19.584|-0.560|
| unchanged native decode1full |16.960|16.960|0.000|
| unchanged Gluon verify16full |139.056|139.392|+0.512|

The unchanged Gluon offset is retained as variation, not a kernel regression.
Other controls and nearby cases are in the raw JSON. Do not infer paired deltas
by subtracting independent medians, or interpret the local aggregate as H200.

Native-vs-new-main comparisons preserve the small-full advantage: B1/page1
decode17.744->17.040us (paired-0.608), verify18.464->18.256 (-0.224); page64
decode-0.368/verify-0.704us. All eight tested native full cases win, as does
native sliding B1 (-0.416us); unchanged BF16 decode16full control is-0.048us.
This does not prove a universal nearby-shape advantage or erase prior audits.

Precision review: the earlier NOTES session2 FP32-only-scratch assertion had no
cited organizer clarification. TASK and the [published rules](https://openkernelchallenge.dev/rules)
state that activations remain BF16 where SGLang keeps them BF16 and constrain
cache/query precision; neither explicitly names split scratch. This integration
follows main's interpretation while retaining FP32 accumulation and native
interfaces. This is not an organizer ruling or a claim that numerical tolerance
alone establishes compliance. No H200 check or scored submission was requested.

Reproduce from repo root with the existing venv:

```sh
python profiles/session36-integration/validate.py suite
python profiles/session36-integration/validate.py compare
python profiles/session36-integration/validate.py fresh
python profiles/session36-integration/compare_main.py compare
python profiles/session36-integration/audit.py
```

The current source needs its own exact-source H200 qualification before any
future scored submission. Older Codex checks and main's older score do not qualify it.

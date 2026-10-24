# Session 43: native full-merge load issuance

Rejected; no production changes and no judge requests. This archive pins the
pre-9a43323 production source (SHA ee782c4d69ed). The separate late-main
integration is not covered by these timings.

`generate.py` moves the independent accumulator loads before consumption of the
max/sum loads, keeping all reads after the PDL dependency wait. FP32 arithmetic,
layout, dispatch, masks and launch configuration are unchanged. Only the native
full-merge definition and its extension name differ.

Assembly disproves the proposed serial-round-trip mechanism: production already
issues 16 of 17 coherent loads before its first maximum in PDL mode, and all 17
before the first exp2. The candidate issues all 17 before the maximum, but still
has 544 static instruction sites. Non-PDL: both have 14 loads before the maximum,
all 17 before exp2, and 536 instruction sites. Both modes retain 48 registers,
3200 reported shared bytes, zero stack/local allocation. Both PDL versions keep
all 17 coherent loads after ACQBULK. These are static scheduling observations,
not measured hardware stall durations.

H100 complete-graph paired candidate-minus-parent deltas (microseconds):

| Full case | initial | second seed, reversed capture |
|---|---:|---:|
| B1 L1 80K page1 | -0.016 | -0.080 |
| B1 L8 80K page1 | -0.192 | +0.112 |
| B1 L1 80K page64 | +0.032 | -0.016 |
| B1 L8 80K page64 | +0.064 | -0.208 |
| B2 L8 9K page64 scaled edges | -0.048 | -0.032 |
| B2 L8 9K page1 scaled edges | +0.016 | +0.096 |

Two unchanged initial controls are +0.032/+0.064us. The scored verify saving
reverses; no repeatable material improvement or new dispatch is established.
Do not extend this into another instruction-placement sweep.

Validation: candidate 16/16 then all 12 benchmark shapes (local aggregate 2.020x
against fixed SGLang/H200 numbers; not H200 qualification). Eight initial and
six confirmation cases, 80 adjacent AB/BA pairs each: 1120 pairs and 84 independent
output/cache checks. Eight cases, four fresh inputs, two sources: 64 fresh graph
replays, including changed QKV/cache/routes/lengths/positions/cache slots and
halved/empty histories. Six cases exercise native full; two are unchanged
controls. Only two standard-suite cases exercise the changed native full path.

The custom driver holds the shared GPU lock across generation, reference,
capture and replay, flushes 512MB before measured graphs, and subtracts an empty
replay. Standard suite holds the lock; standard bench uses its timing-loop lock.
All work ran sequentially. No sanitizer or NCU claim. Arithmetic, addresses and
synchronization are unchanged; fresh graph checks and SASS ordering cover the
load-movement concern. `audit.py` checks generated source, unchanged AST nodes,
assembly ordering/resources, pair medians/counts and fresh checks.

Reproduce with the repository venv:

```sh
python experiments/merge_load_issue/generate.py
python experiments/merge_load_issue/disassemble.py
python experiments/merge_load_issue/validate.py suite
python experiments/merge_load_issue/validate.py compare
python experiments/merge_load_issue/validate.py fresh
python experiments/merge_load_issue/validate.py confirm
python experiments/merge_load_issue/audit.py
```

Redirect each validation mode to its corresponding `.log` before running the
audit. `previous`/`merged` in the archived JSON mean parent/candidate.

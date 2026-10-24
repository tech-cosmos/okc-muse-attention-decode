"""Shared-memory bank conflicts per SASS instruction from an ncu SourceCounters csv dump: conf.py dump.csv"""
import csv, sys
rows = list(csv.reader(open(sys.argv[1])))
h = next(i for i, r in enumerate(rows) if r and r[0] == "Address"); hdr = rows[h]; ix = {k: hdr.index(k) for k in hdr}
num = lambda s: float(s.replace(",", "") or 0)
out = []
for r in rows[h + 1:]:
    if len(r) != len(hdr) or r[0] == "Address": continue
    ex = num(r[ix["L1 Wavefronts Shared Excessive"]])
    if ex > 0:
        out.append((ex, r[1], num(r[ix["L1 Wavefronts Shared"]]), num(r[ix["L1 Wavefronts Shared Ideal"]]), num(r[ix["Instructions Executed"]])))
out.sort(reverse=True)
print("total excessive", sum(o[0] for o in out))
for o in out[:20]: print(f"{o[0]:10.0f} {o[1][:50]:50s} wf {o[2]:.0f} ideal {o[3]:.0f} inst {o[4]:.0f}")

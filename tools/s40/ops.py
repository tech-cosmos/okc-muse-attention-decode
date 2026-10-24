"""Aggregate an ncu --page source --print-source sass --csv dump by opcode: executed instructions and stall samples.
  python tools/s40/ops.py dump.csv"""
import csv, sys, collections
rows = list(csv.reader(open(sys.argv[1])))
h = next(i for i, r in enumerate(rows) if r and r[0] == "Address")
hdr = rows[h]
ix = {k: hdr.index(k) for k in hdr}
ex = collections.Counter(); smp = collections.Counter(); st = collections.defaultdict(collections.Counter)
stalls = [k for k in hdr if k.startswith("stall_") and "Not Issued" not in k]
num = lambda s: float(s.replace(",", "") or 0)
for r in rows[h + 1:]:
    if len(r) != len(hdr) or r[0] == "Address":
        continue
    op = r[1].split()[0] if r[1].split() else "?"
    if op.startswith("@"):
        op = r[1].split()[1]
    op = op.split(".")[0]
    ex[op] += num(r[ix["Instructions Executed"]])
    smp[op] += num(r[ix["Warp Stall Sampling (All Samples)"]])
    for k in stalls:
        st[op][k] += num(r[ix[k]])
tot = sum(ex.values()); ts = sum(smp.values())
print(f"total inst {tot:.3e} samples {ts:.0f}")
for op, n in ex.most_common(40):
    top = ", ".join(f"{k[6:]} {v / max(1, smp[op]) * 100:.0f}%" for k, v in st[op].most_common(3) if v)
    print(f"{op:10s} inst {n / tot * 100:5.1f}%  samples {smp[op] / ts * 100:5.1f}%  [{top}]")

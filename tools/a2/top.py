"""agent2: hot SASS lines from an ncu SourceCounters csv. python tools/a2/top.py file.csv [frac]"""
import csv, sys
rows = list(csv.reader(l for l in open(sys.argv[1]) if l.startswith('"')))
hdr = rows[1]; data = rows[2:]
ia, isrc, iss = hdr.index("Address"), hdr.index("Source"), hdr.index("Warp Stall Sampling (All Samples)")
st = [i for i, h in enumerate(hdr) if h.startswith("stall_") and "Not Issued" not in h]
tot = sum(float(r[iss] or 0) for r in data)
fr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
print("total samples", tot)
for r in data:
    v = float(r[iss] or 0)
    if v >= fr * tot:
        top = sorted(((float(r[i] or 0), hdr[i][6:]) for i in st), reverse=True)[:2]
        print(f"{r[ia]:>6} {100*v/tot:5.1f}% {r[isrc][:60]:60s} " + " ".join(f"{n}:{x:.0f}" for x, n in top if x))

import csv, collections, sys
rows=list(csv.reader(open(sys.argv[1])))
for k,r in enumerate(rows):
    if r and r[0]=="Address": h=r; start=k+1; break
idx={n:i for i,n in enumerate(h)}
data=[r for r in rows[start:] if len(r)==len(h)]
tot=collections.Counter(); samp=collections.Counter()
for r in data:
    src=r[idx["Source"]].split()
    if not src: continue
    op=src[1] if src[0].startswith("@") else src[0]
    base=op if len(sys.argv)>2 else op.split('.')[0]
    tot[base]+=int(r[idx["Instructions Executed"]] or 0); samp[base]+=int(r[idx["Warp Stall Sampling (All Samples)"]] or 0)
T=sum(tot.values()); S=sum(samp.values())
print("total", T)
for k,v in tot.most_common(25): print(f"{k:28s} {v:10d} {v/T*100:5.1f}%  samples {samp[k]/S*100:5.1f}%")

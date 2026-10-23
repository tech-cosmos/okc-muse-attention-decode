import csv,sys
rows=list(csv.reader(open(sys.argv[1])))
k=[i for i,r in enumerate(rows) if r and r[0]=="Address"][0]
h=[x.strip() for x in rows[k]]; data=[r for r in rows[k+1:] if len(r)==len(h)]
ia=h.index("Address"); isrc=h.index("Source"); iss=h.index("Warp Stall Sampling (All Samples)")
stall_cols=[i for i,x in enumerate(h) if x.startswith("stall_")]
tot=sum(float(r[iss] or 0) for r in data)
print("total samples",tot, "instrs", len(data))
for r in data:
    s=float(r[iss] or 0)
    if s>=tot*float(sys.argv[2] if len(sys.argv)>2 else 0.012):
        top=sorted(((float(r[i] or 0),h[i][6:]) for i in stall_cols),reverse=True)[:2]
        print(r[ia], f"{s:6.0f}", r[isrc][:58], [(int(a),b) for a,b in top])

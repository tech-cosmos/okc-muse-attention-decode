"""Batch-1 full plan variants, whole call, cold (judge flush) vs warm (no flush: 20 MB K/V L2-resident on the H100).
Warm approximates the latency-chain regime the H200 is suspected to be in (its batch-1 full times barely beat the H100's).
usage: python tools/s113/d1warm.py decode|verify '<json list of plan overrides>' [rounds]"""
import sys, json, torch; sys.path.insert(0, "tools"); sys.path.insert(0, ".")
import submission, bench
load = sys.argv[1] + " 1" if " " not in sys.argv[1] else sys.argv[1]
plans = json.loads(sys.argv[2]); rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
_, c = [x for x in bench.SCORED if x[0].split() == load.split() and x[1]["layer"] == "full"][0]
lname = [x for x in bench.SCORED if x[0].split() == load.split() and x[1]["layer"] == "full"][0][0]
big = torch.empty(512 << 20, dtype=torch.uint8, device="cuda"); small = torch.empty(8, dtype=torch.uint8, device="cuda")
res = {(i, m): [] for i in range(len(plans)) for m in ("cold", "warm")}
for r in range(rounds):
    for mode in ("cold", "warm"):
        bench._FLUSH = big if mode == "cold" else small
        for i, p in enumerate(plans):
            res[(i, mode)].append(bench.run_shape(lname, c, p or None))
import statistics as st
for i, p in enumerate(plans):
    print(json.dumps(p), "cold %.2f warm %.2f" % (st.median(res[(i, "cold")]), st.median(res[(i, "warm")])), flush=True)

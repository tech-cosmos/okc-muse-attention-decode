"""Compute-bound check for full layers: time each shape as is and with every page-table entry folded into a few
thousand rows (all K/V reads hit L2: what's left is the kernel's own work), on all SMs and under tools/emu.py's
H200 per-SM bandwidth emulation (BLOCK SMs parked).  python tools/s19/cbound.py [block=40] [filter=full] [plan json]"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import emu, bench, submission, torch  # noqa: E402

block = int(sys.argv[1]) if len(sys.argv) > 1 else 40
filt = sys.argv[2] if len(sys.argv) > 2 else "full"
plan = json.loads(sys.argv[3]) if len(sys.argv) > 3 else None
orig = submission._plan
if plan:
    submission._plan = lambda cfg, mp, dev: {**orig(cfg, mp, dev), **plan}
x = torch.zeros(1, device="cuda")
for blk in (0, block):
    submission._SMS[0] = 132 - blk
    empty = emu.time_blocked(lambda: x.add_(1), blk, reps=60)
    for load, c in bench.SCORED:
        if filt not in f"{load} {c['layer']}":
            continue
        data = bench.data_for(load, c)
        t = emu.time_blocked(lambda: submission.kernel_fn(data), blk, 20) - empty
        d2 = dict(data)
        d2["page_table"] = data["page_table"] % 4096
        t2 = emu.time_blocked(lambda: submission.kernel_fn(d2), blk, 20) - empty
        print(f"sms {132 - blk:3d} {load:10s} {c['layer']:8s} real {t:8.2f}  L2-resident {t2:8.2f}", flush=True)

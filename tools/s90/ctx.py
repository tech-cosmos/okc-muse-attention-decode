"""Verify/decode 1 full attention-only time vs ctx (same plan), cold and warm: fixed vs per-key cost."""
import sys, os, json, torch; sys.path.insert(0, "tools"); sys.path.insert(0, ".")
import submission, bench
class Nop:
    def __getitem__(self, g): return lambda *a, **k: None
load = sys.argv[1]; plan = json.loads(sys.argv[2]) if len(sys.argv) > 2 else None
load_, c0 = [x for x in bench.SCORED if x[0] == load and x[1]["layer"] == "full"][0]
submission._combine = Nop()
for ctx in (10000, 20000, 40000, 80000):
    c = dict(c0, ctx=ctx)
    out = []
    for mode in ("cold", "warm"):
        bench._FLUSH = torch.empty((512 << 20) if mode == "cold" else 8, dtype=torch.uint8, device="cuda")
        out.append(bench.run_shape(f"{load_}-{ctx}", c, plan))
    print(load, ctx, "cold %.2f warm %.2f" % tuple(out), flush=True)

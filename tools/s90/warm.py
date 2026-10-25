"""Batch-1 full: whole / attention-only, cold (judge-like flush) vs warm (no flush: K/V in L2 where it fits)."""
import sys, os, torch; sys.path.insert(0, "tools"); sys.path.insert(0, ".")
import submission, bench
class Nop:
    def __getitem__(self, g): return lambda *a, **k: None
orig = submission._combine
for load in sys.argv[1:]:
    load_, c = [x for x in bench.SCORED if x[0] == load and x[1]["layer"] == "full"][0]
    for mode in ("cold", "warm", "cold", "warm"):
        bench._FLUSH = torch.empty((512 << 20) if mode == "cold" else 8, dtype=torch.uint8, device="cuda")
        submission._combine = orig
        w = bench.run_shape(load_, c, None)
        submission._combine = Nop()
        a = bench.run_shape(load_, c, None)
        submission._combine = orig
        print(load, mode, "whole %.2f attn %.2f" % (w, a), flush=True)

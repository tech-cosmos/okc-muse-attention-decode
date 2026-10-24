"""Time a scored shape with plan overrides, with and without the merge launch (no-op merge = wrong results).
usage: python tools/s23/abl.py "decode 1 full" '<json list of plan overrides>'"""
import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import bench, submission

class _Nop:
    def __getitem__(self, grid):
        return lambda *a, **k: None

shape = sys.argv[1]
plans = json.loads(sys.argv[2])
for load, c in bench.SCORED:
    if f"{load} {c['layer']}" != shape:
        continue
    for p in plans:
        full = bench.run_shape(load, c, p)
        saved = submission._combine, submission._combine_hb
        submission._combine = submission._combine_hb = _Nop()
        try:
            attn = bench.run_shape(load, c, p)
        finally:
            submission._combine, submission._combine_hb = saved
        print(f"{full:7.2f} total {attn:7.2f} no-merge  {p}", flush=True)

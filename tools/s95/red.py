"""plan red (decode 1 full in-grid atomic merge): correctness vs main plan and reference, then timing.
argv[1]: 'c' correctness only, 't' timing (paired rounds)."""
import sys, os; sys.path.insert(0, "tools"); sys.path.insert(0, ".")
import torch, submission, reference, bench
orig = submission._plan
def setp(extra):
    submission._plan = (lambda cfg, mp, dev: {**orig(cfg, mp, dev), **extra}) if extra else orig
mode = sys.argv[1] if len(sys.argv) > 1 else "c"
if "c" in mode:
    cases = [dict(seed=s, batch=1, ctx=ctx, heads="tp8", layer="full", page=1, q_len=1, dist=dist)
             for s, ctx, dist in [(1, 80000, "coherent"), (2, 80000, "soft"), (3, 5000, "normal"), (4, 100000, "edges"),
                                  (5, 300, "coherent")]]
    cases.append(dict(seed=6, batch=1, ctx=20000, heads="tp8", layer="full", page=64, q_len=1, dist="soft"))
    for c in cases:
        for ks in (None, 40.0, 1e-3):
            data = reference.generate_input(**c)
            if ks is not None:  # push scores out of [-60, 100] (fallback path) or squash them
                data["k_scale"] = data["k_scale"] * ks
            outs = []
            for extra in ({"red": False}, {"red": True}):
                setp(extra)
                for _ in range(3):  # replays must stay consistent (sums/counters re-zeroed)
                    d2 = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in data.items()}
                    o = submission.kernel_fn(d2); torch.cuda.synchronize()
                outs.append(o.float())
                ok, msg = reference.check_implementation(d2, reference.collect(d2, o))
            setp({})
            cnt = submission._COUNTERS.get((outs[0].device, "r"))
            print(c["seed"], c["ctx"], c["dist"], ks, "ref ok" if ok else "REF FAIL " + str(msg)[:200],
                  "max|red-main| %.4g" % (outs[1] - outs[0]).abs().max().item(),
                  "cnt", None if cnt is None else cnt[:3].tolist(), flush=True)
if "t" in mode:
    load, c = bench.SCORED[1]
    for _ in range(int(os.environ.get("R", 5))):
        print("main %.2f red %.2f" % (bench.run_shape(load, c, {"red": False}), bench.run_shape(load, c, {"red": True})), flush=True)

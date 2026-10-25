"""s101: exp/vhring (plan nh) vs default at verify 16/32 full: max |diff|, then paired real / all-L2 times."""
import sys, statistics; sys.path.insert(0, "tools"); sys.path.insert(0, ".")
import torch, bench, submission, reference
orig = submission._plan
variants = [dict(), dict(nh=2, stages=4)] + [eval(a) for a in sys.argv[2:]]
for load in sys.argv[1].split(","):
    _, c = [x for x in bench.SCORED if x[0] == load and x[1]["layer"] == "full"][0]
    base = bench.data_for(load, c)
    for l2 in (False, True):
        d = dict(base)
        if l2: d["page_table"] = d["page_table"] % 4096
        outs, ts = [], {i: [] for i in range(len(variants))}
        for i, v in enumerate(variants):
            submission._plan = lambda cfg, mp, dev, v=v: {**orig(cfg, mp, dev), **v}
            outs.append(submission.kernel_fn(d)); torch.cuda.synchronize()
        if not l2:
            ok, msg = reference.check_implementation(d, reference.collect(d, outs[1]))
            print(load, "nh check:", ok, msg[:80], " max|nh-base|", (outs[1].float() - outs[0].float()).abs().max().item())
        for r in range(5):
            for i, v in enumerate(variants):
                submission._plan = lambda cfg, mp, dev, v=v: {**orig(cfg, mp, dev), **v}
                ts[i].append(bench.time_graph(lambda: submission.kernel_fn(d), reps=20))
        submission._plan = orig
        print(load, "L2" if l2 else "real", " ".join(f"{variants[i]}: {statistics.median(t):.1f}" for i, t in ts.items()), flush=True)

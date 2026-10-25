"""s83: TV variants of _gw_verify at verify 16/32 full: correctness vs reference, then paired timing real + all-L2.
tv.py rounds '{"tv":1}' '{"tv":2,"stages":4}' ..."""
import json, statistics, sys
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench, reference, submission, torch
rounds = int(sys.argv[1]); variants = [{}] + [json.loads(a) for a in sys.argv[2:]]
orig = submission._plan
def run(d, v):
    submission._plan = lambda cfg, mp, dev: {**orig(cfg, mp, dev), **v}
    try: return submission.kernel_fn(d)
    finally: submission._plan = orig
for batch in (32, 16):
    c = dict(batch=batch, ctx=80000, heads="tp8", layer="full", page=1, q_len=8, dist="coherent", spread="agentic")
    base = reference.generate_input(seed=99, **c)
    for v in variants:
        out = run(base, v); torch.cuda.synchronize()
        ok, msg = reference.check_implementation(base, reference.collect(base, out))
        print(batch, v, "PASS" if ok else "FAIL", msg[:100], flush=True)
    for l2 in (False, True):
        d = dict(base)
        if l2: d["page_table"] = d["page_table"] % 4096
        ts = [[] for _ in variants]
        for r in range(rounds):
            order = list(range(len(variants)))
            if r % 2: order.reverse()
            for k in order: ts[k].append(bench.time_graph(lambda: run(d, variants[k])))
        print(f"B={batch} {'L2  ' if l2 else 'real'} " + "  ".join(
            f"{json.dumps(v)} {statistics.median(t):.2f} ({statistics.median([y - x for x, y in zip(ts[0], t)]):+.2f})"
            for v, t in zip(variants, ts)), flush=True)

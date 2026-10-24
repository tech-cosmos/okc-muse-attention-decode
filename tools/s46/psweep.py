"""Paired plan-override sweep at verify full (real + all-L2): psweep.py rounds batches json..."""
import sys, json, statistics; sys.path.insert(0,"/home/daytona/repo/tools"); sys.path.insert(0,"/home/daytona/repo")
import bench, reference, submission
rounds = int(sys.argv[1]); batches = [int(x) for x in sys.argv[2].split(",")]; overs = [json.loads(x) for x in sys.argv[3:]]
orig = submission._plan
def run(d, o):
    submission._plan = (lambda cfg, mp, dev: {**orig(cfg, mp, dev), **o}) if o else orig
    try: return bench.time_graph(lambda: submission.kernel_fn(d))
    finally: submission._plan = orig
for b in batches:
    c = dict(batch=b, ctx=80000, heads="tp8", layer="full", page=1, q_len=8, dist="coherent", spread="agentic")
    base = reference.generate_input(seed=99, **c)
    for l2 in (False, True):
        d = dict(base)
        if l2: d["page_table"] = base["page_table"] % 4096
        for o in overs:
            ta, tb = [], []
            for _ in range(rounds):
                ta.append(run(d, None)); tb.append(run(d, o))
            df = [y - x for x, y in zip(ta, tb)]
            print(f"B={b} {'L2 ' if l2 else 'real'} base {statistics.median(ta):7.2f} new {statistics.median(tb):7.2f} paired {statistics.median(df):+.2f} wins {sum(x<0 for x in df)}/{rounds} {o}", flush=True)

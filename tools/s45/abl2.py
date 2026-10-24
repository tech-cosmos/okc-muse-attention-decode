"""Paired A/B of two submission files at verify/decode full, real and all-L2 (page table % 4096).
abl2.py a.py b.py rounds batch q_len"""
import importlib.util, statistics, sys
sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench, reference, torch
def load(path, name):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
A, B = load(sys.argv[1], "sub_a"), load(sys.argv[2], "sub_b")
rounds, batch, ql = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
c = dict(batch=batch, ctx=80000, heads="tp8", layer="full", page=1, q_len=ql, dist="coherent",
         **({"spread": "agentic"} if batch > 1 else {}))
base = reference.generate_input(seed=99, **c)
for l2 in (False, True):
    d = dict(base)
    if l2:
        d["page_table"] = base["page_table"] % 4096
    ta, tb = [], []
    for _ in range(rounds):
        ta.append(bench.time_graph(lambda: A.kernel_fn(d)))
        tb.append(bench.time_graph(lambda: B.kernel_fn(d)))
    df = [y - x for x, y in zip(ta, tb)]
    print(f"B={batch} L={ql} {'L2  ' if l2 else 'real'} A {statistics.median(ta):7.2f} B {statistics.median(tb):7.2f} "
          f"paired {statistics.median(df):+.2f} wins {sum(x < 0 for x in df)}/{rounds}", flush=True)

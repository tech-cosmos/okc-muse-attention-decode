import sys, fcntl; d=sys.argv[1]; sys.path.insert(0, d); sys.path.insert(1, "/home/daytona/repo/tools"); sys.path.insert(2, "/home/daytona/repo")
import submission, bench, torch, json, os
P=json.loads(os.environ.get('PLAN','{}'))
orig=submission._plan
submission._plan=lambda c,m,d:{**orig(c,m,d),**P}
for l, c in bench.SCORED:
    if l == "verify 16" and c["layer"] == "full":
        data = bench.data_for(l, c)
        submission.kernel_fn(data); torch.cuda.synchronize()
        with open("/tmp/okc-gpu.lock", "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            submission.kernel_fn(data); torch.cuda.synchronize()

"""agent2 s14: per-CTA %globaltimer stamps of nvs cl mode (NV_STAMP 1), verify 32 sliding, cold L2.
  python tools/a2/clstamp.py MODE"""
import sys, torch, importlib, statistics
sys.path.insert(0, "/home/daytona/repo2/tools"); sys.path.insert(0, "/tmp")
src = open("/home/daytona/repo2/submission.py").read().replace("#define NV_HALF 1\n", "#define NV_HALF 1\n#define NV_STAMP 1\n", 1)
src = src.replace('name="muse_nvs_', 'name="muse_nvsstamp_')
open("/tmp/a2clstamp_sub.py", "w").write(src)
import bench
sub = importlib.import_module("a2clstamp_sub")
pass
for l, c in bench.SCORED:
    if l == "verify 32" and c["layer"] == "sliding":
        data = bench.data_for(l, c)
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
res = []
for it in range(6):
    flush.zero_(); torch.cuda.synchronize()
    sub.kernel_fn(data); torch.cuda.synchronize()
    st = sub._LAST_PML.flatten().view(torch.int64)[:128 * 8].view(128, 8).cpu()
    t0 = st[:, 0].min().item()
    r = (st[:, :7] - t0).float() / 1000
    if it >= 2: res.append(r)
r = torch.stack(res).median(0).values
r0, r1 = r[0::2], r[1::2]
names = ["start", "loopend h0", "loopend h1", "inCTA merged", "push/wait done", "end", "merged2"]
for k, n in enumerate(names):
    a, b = r0[:, k], r1[:, k]
    print(f"{n:16s} rank0 med {a.median():6.2f} max {a.max():6.2f} | rank1 med {b.median():6.2f} max {b.max():6.2f}")
if len(sys.argv) > 2:
    st = sub._LAST_PML.flatten().view(torch.int64)[:128 * 8].view(128, 8).cpu()
    print(sorted(set(((st[:, :6] - st[:, :6].min()) % 4096).flatten().tolist()))[:40])

import sys, torch, statistics
sys.path.insert(0, "/home/daytona/repo2/tools"); sys.path.insert(0, "/home/daytona/repo2")
import bench, submission
load = sys.argv[1]
cap = {}
orig = torch.empty
def emp(*a, **k):
    t = orig(*a, **k)
    if t.dim() == 5 and t.dtype == torch.float32 and t.shape[-1] == 64: cap['pacc'] = t
    return t
for l, c in bench.SCORED:
    if l != load or c["layer"] != "sliding": continue
    submission.NVS_MIN_BATCH = 16
    data = bench.data_for(l, c)
    fl = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    for _ in range(3): submission.kernel_fn(data)
    torch.empty = emp; fl.zero_(); submission.kernel_fn(data); torch.cuda.synchronize(); torch.empty = orig
    p = cap['pacc']; B, H, S = p.shape[:3]
    t = p[:, :, :, 0, :10].contiguous().view(torch.int64)[..., :5].double()  # [B, H, S, 5]
    z = t[..., 0].min()
    t = (t - z) / 1000
    names = ["start", "q done", "tile0 ready", "loop done", "stored"]
    for i, nm in enumerate(names):
        a = t[..., i].flatten(); lst = t[:, :, S - 1, i].flatten(); oth = t[:, :, :S - 1, i].flatten()
        print(f"{nm:12s} med {a.median():6.2f} max {a.max():6.2f} | last split med {lst.median():6.2f} max {lst.max():6.2f} | others med {oth.median():6.2f} max {oth.max():6.2f}")

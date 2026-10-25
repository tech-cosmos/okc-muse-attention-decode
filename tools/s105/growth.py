"""s105: how far does a full-layer row's running max grow past the max of a segment's first 64-key tile?
For every 64-key tile start t of each request, growth = max(scores in tiles t..t+W-1) - max(scores in tile t), in
binades (log2 units of P). Prints max growth and the fraction of (row, start) with growth > 15 (FP16 P overflow risk)."""
import sys, torch
sys.path.insert(0, "/home/daytona/repo"); sys.path.insert(0, "/home/daytona/repo/tools")
import reference as R, bench
W = int(sys.argv[1]) if len(sys.argv) > 1 else 16
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 7
ONLY = len(sys.argv) > 3 and sys.argv[3] == "1"
TH = 30.0
K = int(sys.argv[4]) if len(sys.argv) > 4 else 1
cases = [] if ONLY else [c for c in bench.TESTS if c["layer"] == "full"] if hasattr(bench, "TESTS") else []
cases += [dict(batch=b, ctx=80000, heads="tp8", layer="full", page=1, q_len=l, dist="coherent", spread="agentic")
          for b in ((32,) if ONLY else (1, 16, 32)) for l in ((8,) if ONLY else (1, 8))]
for c in cases:
    d = R.generate_input(seed=SEED, **c)
    cfg = d["config"]; hq, hkv = cfg["num_q_heads"], cfg["num_kv_heads"]
    q = R.split_qkv(d)[0]
    qn = R._rms(q)  # [B, L, Hq, 64]
    kc = d["k_cache"].view(-1, hkv, 64); ks = d["k_scale"].float()
    worst, over, tot = -1e9, 0, 0
    for b in range(cfg["batch"]):
        n = int(d["seq_lens"][b]); n -= n % 64
        if n < 64: continue
        rows = R._rows({"page_table": d["page_table"][b:b+1], "config": cfg}, torch.arange(n, device="cuda")[None])[0]
        k = kc[rows].float() * ks  # [n, hkv, 64]
        g = hq // hkv
        qq = qn[b].view(-1, hkv, g, 64)  # [L, hkv, g, 64]
        s = torch.einsum("lhgd,nhd->lhgn", qq, k) * (0.125 * 1.4426950408889634 * R.QK_SCALE_FACTOR)
        tm = s.view(*s.shape[:3], -1, 64).amax(-1)  # tile maxima [L, hkv, g, T]
        T = tm.shape[-1]
        fw = torch.stack([tm[..., i:i+W].amax(-1) if i + W <= T else tm[..., i:].amax(-1) for i in range(T)], -1)
        ref = torch.stack([tm[..., i:i+K].amax(-1) for i in range(T)], -1)
        gr = fw - ref
        worst = max(worst, gr.max().item()); over += (gr > TH).sum().item(); tot += gr.numel()
    print(f"{c} W={W}: max growth {worst:.2f} binades (TH {TH}), frac>TH {over / max(tot, 1):.2e}", flush=True)

"""s105: adversarial verify-full inputs forcing FP16 P overflow past the first tile (late keys = A x the first query
head's normalized query: ~11.5 * A binades of score), so LZ 7's non-finite check + _combine_bal REDO must kick in.
Run from a dir whose submission.py is the candidate: python adv.py [A ...]."""
import sys, torch
sys.path.insert(0, "."); sys.path.insert(0, "/home/daytona/repo")
import reference as R, submission
for A in [float(x) for x in sys.argv[1:]] or [1.0, 2.0, 3.0]:
    for c in (dict(batch=4, ctx=6000, heads="tp8", layer="full", q_len=8),
              dict(batch=2, ctx=20000, heads="dp", layer="full", q_len=8, dist="soft"),
              dict(batch=32, ctx=8000, heads="tp8", layer="full", q_len=8, page=1)):
        d = R.generate_input(seed=11, **c)
        cfg = d["config"]; hq, hkv = cfg["num_q_heads"], cfg["num_kv_heads"]
        qn = R._rms(R.split_qkv(d)[0])  # [B, L, Hq, 64]
        kc = d["k_cache"].view(-1, hkv, 64)
        for b in range(cfg["batch"]):
            n = int(d["seq_lens"][b])
            pos = torch.arange(n // 3, n, 97, device="cuda")  # sparse late keys in many segments
            rows = R._rows({"page_table": d["page_table"][b:b+1], "config": cfg}, pos[None])[0]
            dirn = qn[b, 0, :: hq // hkv]  # [Hkv, 64]
            kc[rows] = (A * dirn / d["k_scale"]).clamp(-448, 448).to(kc.dtype)[None].expand(len(rows), -1, -1)
        out = submission.kernel_fn(d); torch.cuda.synchronize()
        ok, msg = R.check_implementation(d, R.collect(d, out))
        print(f"A={A} {c}: {'PASS' if ok else 'FAIL'} {msg[:80]}", flush=True)

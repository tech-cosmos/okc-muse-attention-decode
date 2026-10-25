"""s118: decode 1 full (red path) vs reference, at _red_fin warps 4 and 16, several seeds (incl. the scored shape)."""
import sys; sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench, reference, submission, torch
load, c = [x for x in bench.SCORED if f"{x[0]} {x[1]['layer']}" == "decode 1 full"][0]
for seed in (1, 2, 3, 99):
    outs = []
    for nw in (4, 16):
        submission.RF_NW1 = nw
        data = reference.generate_input(seed=seed, **c)
        out = submission.kernel_fn(data); torch.cuda.synchronize()
        outs.append(out.float().clone())
        ok, msg = reference.check_implementation(data, reference.collect(data, out))
        print(seed, nw, ok, str(msg)[:80])
    print("  max |16 - 4|", (outs[0] - outs[1]).abs().max().item())

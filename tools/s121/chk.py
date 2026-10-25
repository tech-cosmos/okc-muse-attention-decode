"""s121: verify 1 full on agent2's nvs (window 0) vs reference, several S and seeds."""
import sys; sys.path.insert(0, "/home/daytona/repo/tools"); sys.path.insert(0, "/home/daytona/repo")
import bench, reference, submission, torch
load, c = [x for x in bench.SCORED if f"{x[0]} {x[1]['layer']}" == "verify 1 full"][0]
for seed in (1, 2, 99):
    data = reference.generate_input(seed=seed, **c)
    submission.NVS_FULL1 = False
    base = submission.kernel_fn(data).float().clone()
    for S in (32, 64, 128):
        submission.NVS_FULL1, submission.NVS_F1_S = True, S
        out = submission.kernel_fn(data); torch.cuda.synchronize()
        ok, msg = reference.check_implementation(data, reference.collect(data, out))
        print(seed, S, ok, str(msg)[:80], "max|nvs-main|", (out.float() - base).abs().max().item(), flush=True)

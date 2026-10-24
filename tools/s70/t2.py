import sys, json; sys.path.insert(0, "."); sys.path.insert(0, "tools")
import torch, reference, submission
over = json.loads(sys.argv[1]); shapes = json.loads(sys.argv[2])
orig = submission._plan
submission._plan = lambda cfg, mp, dev: {**orig(cfg, mp, dev), **over}
for c in shapes:
    data = reference.generate_input(seed=5, **c)
    out = submission.kernel_fn(data); torch.cuda.synchronize()
    print(c, reference.check_implementation(data, reference.collect(data, out)), flush=True)

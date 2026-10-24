"""Bit-identity of a plan override vs the default plan at verify 16/32 full: ident.py json"""
import sys; sys.path.insert(0,"/home/daytona/repo/tools"); sys.path.insert(0,"/home/daytona/repo")
import reference, submission, torch, json
over = json.loads(sys.argv[1])
orig = submission._plan
submission._plan = lambda cfg, mp, dev: {**orig(cfg, mp, dev), **over}
for b in (16, 32):
    c = dict(batch=b, ctx=80000, heads="tp8", layer="full", page=1, q_len=8, dist="coherent", spread="agentic")
    d = reference.generate_input(seed=7, **c)
    o = submission.kernel_fn(d).float()
    o0 = (lambda: (setattr(submission,'_plan',orig), submission.kernel_fn(d).float())[1])()
    submission._plan = lambda cfg, mp, dev: {**orig(cfg, mp, dev), **over}
    print(b, "max diff vs main plan", (o-o0).abs().max().item())

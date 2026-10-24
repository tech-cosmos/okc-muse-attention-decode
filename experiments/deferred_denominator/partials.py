"""Compare native FP32 partials, including a long recurrence, on fresh inputs."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
import reference

ROOT = Path(__file__).resolve().parent

def module(variant):
    os.environ['TDEC_VARIANT'] = variant
    spec = importlib.util.spec_from_file_location('partial_' + variant, ROOT / 'native.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def run():
    mods = {name: module(name) for name in ('natural', 'deferred', 'lane')}
    results = []
    for page in (1, 64):
        for ctx in (63, 8192, 80000, 1048576):
            for scale in (1., 1 / 64):
                case = dict(batch=1, ctx=ctx, heads='tp8', layer='full', page=page,
                            q_len=1, kv_scale=scale, dist='coherent')
                d = reference.generate_input(seed=36400 + len(results), **case)
                baseline = mods['natural'].partials(d)
                baseline_cache = [d[k].view(torch.uint8).clone() for k in ('k_cache', 'v_cache')]
                for name in ('deferred', 'lane'):
                    a, ml = mods[name].partials(d)
                    base_a, base_ml = baseline
                    assert torch.isfinite(a).all() and torch.isfinite(ml).all()
                    # This mechanism changes only denominator association. Require
                    # exact unchanged numerator/maxima/cache on these comparisons.
                    assert torch.equal(a, base_a), (case, name, 'numerator')
                    assert torch.equal(ml[..., 0], base_ml[..., 0]), (case, name, 'maxima')
                    relative = ((ml[..., 1] - base_ml[..., 1]).abs() /
                                base_ml[..., 1].clamp_min(1e-20)).max().item()
                    assert relative < 1e-5, (case, name, relative)
                    for key, expected in zip(('k_cache', 'v_cache'), baseline_cache):
                        assert torch.equal(d[key].view(torch.uint8), expected), (case, name, key)
                    row = dict(shape=case, variant=name, max_relative_denominator_error=relative,
                               exact_numerator=True, exact_maxima=True, exact_cache=True)
                    results.append(row)
                    print(json.dumps(row), flush=True)
                del d, baseline, baseline_cache, a, ml, base_a, base_ml
                torch.cuda.empty_cache()
    (ROOT / 'partials.json').write_text(json.dumps(results, indent=2) + '\n')

if __name__ == '__main__':
    with open('/tmp/okc-gpu.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run()

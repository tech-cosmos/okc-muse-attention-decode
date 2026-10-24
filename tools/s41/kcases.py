"""tools/s19/cases.py with plan overrides: python tools/s41/kcases.py '{"kst": true}'"""
import json, os, runpy, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import submission  # noqa: E402
ov = json.loads(sys.argv[1])
orig = submission._plan
submission._plan = lambda cfg, mp, dev: {**orig(cfg, mp, dev), **ov}
runpy.run_path(os.path.join(os.path.dirname(__file__), "../s19/cases.py"), run_name="__main__")

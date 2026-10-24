"""gluon_cases.py with plan overrides from argv[1] (JSON) applied to the Gluon path."""
import sys, os, json, runpy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import submission
P = json.loads(sys.argv[1]); orig = submission._plan
submission._plan = lambda c, m, d: (lambda p: {**p, **P} if p.get("gluon") else p)(orig(c, m, d))
sys.argv = sys.argv[:1]
runpy.run_path(os.path.join(os.path.dirname(__file__), "..", "gluon_cases.py"), run_name="__main__")

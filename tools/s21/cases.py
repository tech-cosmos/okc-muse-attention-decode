"""tools/gluon_cases.py with a plan override: python tools/s21/cases.py '{"gbal": true}'"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import submission  # noqa: E402
plan = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
orig = submission._plan
submission._plan = lambda cfg, mp, dev: ({**orig(cfg, mp, dev), **plan} if orig(cfg, mp, dev).get("gluon") else
                                         orig(cfg, mp, dev))
sys.argv = sys.argv[:1]
exec(open(os.path.join(os.path.dirname(__file__), "../gluon_cases.py")).read())

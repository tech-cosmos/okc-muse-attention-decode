"""Serialize full suites, graph benches, fresh replays, sanitizers and profiling."""
import subprocess,sys,os
from pathlib import Path
p=Path(__file__).resolve().parent
jobs=[('validate.py','forced','test-forced.log'),('bench_all.py','forced','bench-forced.log'),
      ('bench_all.py','all','bench-all.log'),('fresh.py','forced','fresh-forced.log'),
      ('fresh.py','all','fresh-all.log'),('run_sanitizers.py','forced','sanitizers.log'),
      ('run_profiles.py','forced','profiles.log')]
for script,variant,log in jobs:
    with (p/log).open('w') as f:
        r=subprocess.run([sys.executable,str(p/script)],env={**os.environ,'VARIANT':variant},stdout=f,stderr=subprocess.STDOUT)
    print(script,variant,r.returncode,flush=True)
    if r.returncode:raise SystemExit(r.returncode)

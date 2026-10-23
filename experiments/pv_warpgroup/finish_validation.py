"""Sequential correctness, full graph benches, fresh replays, sanitizer/profile jobs."""
import os,subprocess,sys
from pathlib import Path
p=Path(__file__).resolve().parent
jobs=[]
for variant in ('swizzle','early_release'):
 for script,kind in [('validate.py','test'),('bench_all.py','bench'),('fresh.py','fresh')]:
  jobs.append((script,variant,kind+'-'+variant+'.log'))
jobs += [('run_sanitizers.py','early_release','sanitizers.log'),
         ('run_profiles.py','early_release','profiles.log')]
for script,variant,log in jobs:
 with (p/'results'/log).open('w') as f:
  result=subprocess.run([sys.executable,str(p/script)],env={**os.environ,'VARIANT':variant},stdout=f,stderr=subprocess.STDOUT)
 print(script,variant,result.returncode,flush=True)
 if result.returncode:raise SystemExit(result.returncode)

"""Serialize reference tests, benches, fresh graphs, sanitizers and final profiles."""
import os,subprocess,sys
from pathlib import Path
p=Path(__file__).resolve().parent
jobs=[]
for variant in ('overlap','latebar','nobar','latebar_nobar'):
 for script,kind in [('validate.py','test'),('bench_all.py','bench'),('fresh.py','fresh')]:
  if script!='fresh.py' or variant in ('nobar','latebar_nobar'):
   jobs.append((script,variant,kind+'-'+variant+'.log'))
jobs += [('run_sanitizers.py','overlap','sanitizers.log'),('run_profiles.py','overlap','profiles.log')]
for script,variant,log in jobs:
 with (p/'results'/log).open('w') as f:
  result=subprocess.run([sys.executable,str(p/script)],env={**os.environ,'VARIANT':variant},stdout=f,stderr=subprocess.STDOUT)
 print(script,variant,result.returncode,flush=True)
 if result.returncode:raise SystemExit(result.returncode)

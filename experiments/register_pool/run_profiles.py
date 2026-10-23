import os,subprocess,sys
from pathlib import Path
p=Path(__file__).resolve().parent
for variant in ('baseline','late2','late3'):
 cmd=['/usr/local/cuda/bin/ncu','--target-processes','all','--kernel-name','regex:verify','--launch-skip','2','--launch-count','1','--section','LaunchStats','--section','Occupancy','--section','SchedulerStats','--section','WarpStateStats',sys.executable,str(p/'profile.py')]
 with (p/('profile-'+variant+'.log')).open('w') as f:
  r=subprocess.run(cmd,env={**os.environ,'VARIANT':variant},stdout=f,stderr=subprocess.STDOUT)
 print(variant,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)

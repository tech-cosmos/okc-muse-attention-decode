import subprocess,sys
from pathlib import Path
p=Path(__file__).resolve().parent
for tool in ('memcheck','synccheck','racecheck'):
 cmd=['/usr/local/cuda/bin/compute-sanitizer','--tool',tool,'--error-exitcode','77',sys.executable,str(p/'sanitize.py')]
 with (p/(tool+'.log')).open('w') as f:
  r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 print(tool,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)

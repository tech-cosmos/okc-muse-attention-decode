import os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
for tool in ('memcheck','synccheck','racecheck'):
 with (root/'results'/f'{tool}.log').open('w') as log:
  cp=subprocess.run(['/usr/local/cuda-13.0/bin/compute-sanitizer','--tool',tool,'--error-exitcode','93',sys.executable,str(root/'synthetic.py')],stdout=log,stderr=subprocess.STDOUT)
 print(tool,cp.returncode,flush=True)
 if cp.returncode:sys.exit(cp.returncode)

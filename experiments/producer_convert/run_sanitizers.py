from pathlib import Path
import subprocess,sys
root=Path('experiments/producer_convert')
for mode in ('memcheck','synccheck','racecheck'):
 with (root/f'{mode}.log').open('w') as f:
  subprocess.run(['/usr/local/cuda-13.0/bin/compute-sanitizer','--tool',mode,'--error-exitcode','99',sys.executable,str(root/'sanitize.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
 print(mode,'complete',flush=True)

import os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
# Sequential to prevent our captures/reference/bench warmups from overlapping.
commands=[
 ('pairs-striped',['pairs.py','--variant','striped']),
 ('validate-warp',['validate.py','--variant','warp']),
 ('validate-striped',['validate.py','--variant','striped']),
 ('synthetic',['synthetic.py']),
]
for name,args in commands:
 with (root/'results'/f'{name}.log').open('w') as log:
  cp=subprocess.run([sys.executable,str(root/args[0]),*args[1:]],stdout=log,stderr=subprocess.STDOUT)
 print(name,cp.returncode,flush=True)
 if cp.returncode:sys.exit(cp.returncode)

from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parent
for batch,page in [(1,1),(8,1),(1,64)]:
    name=f'parent-b{batch}-p{page}'
    cmd=['/usr/local/cuda-13.0/bin/ncu','--kernel-name','regex:sliding','--launch-skip','3','--launch-count','1','--section','SourceCounters','--section','WarpStateStats','--section','Occupancy','--export',str(ROOT/'results'/name),'--force-overwrite','--csv',sys.executable,str(ROOT/'profile.py'),str(batch),str(page)]
    with (ROOT/'results'/f'ncu-{name}.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
    print(name,r.returncode,flush=True)
    if r.returncode:raise SystemExit(r.returncode)
    with (ROOT/'results'/f'ncu-{name}-source.csv').open('w') as f:r=subprocess.run(['/usr/local/cuda-13.0/bin/ncu','--import',str(ROOT/'results'/f'{name}.ncu-rep'),'--page','source','--print-source','sass','--csv'],stdout=f,stderr=subprocess.STDOUT)
    if r.returncode:raise SystemExit(r.returncode)

    with (ROOT/'results'/f'ncu-{name}-raw.csv').open('w') as f:r=subprocess.run(['/usr/local/cuda-13.0/bin/ncu','--import',str(ROOT/'results'/f'{name}.ncu-rep'),'--page','raw','--csv'],stdout=f,stderr=subprocess.STDOUT)
    if r.returncode:raise SystemExit(r.returncode)

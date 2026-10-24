from pathlib import Path
import subprocess,sys,os
ROOT=Path(__file__).resolve().parent
metrics='dram__bytes_read.sum,lts__t_sectors_op_read.sum,lts__t_sector_hit_rate.pct,smsp__inst_executed.sum,launch__registers_per_thread'
for batch,page in [(1,1),(8,1),(8,64)]:
    for name in ['parent','neighbor']:
        dest=ROOT/'results'/f'ncu-{name}-b{batch}-p{page}.log'
        cmd=['/usr/local/cuda-13.0/bin/ncu','--kernel-name','regex:sliding','--launch-skip','3','--launch-count','1','--metrics',metrics,'--csv',sys.executable,str(ROOT/'profile.py'),name,str(batch),str(page)]
        with dest.open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        print(dest.name,r.returncode,flush=True)
        if r.returncode:raise SystemExit(r.returncode)

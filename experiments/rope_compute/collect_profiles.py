"""After timing: compare measured shared-memory bank conflicts, not durations."""
from pathlib import Path
import subprocess,sys,os,fcntl
ROOT=Path(__file__).resolve().parent
metrics='l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,smsp__inst_executed.sum,dram__bytes_read.sum,launch__registers_per_thread'
for label,name,variant in [('parent','parent','candidate'),('computed','candidate','candidate'),('compensated','candidate','compensated')]:
    if len(sys.argv)>1 and label not in sys.argv[1:]:continue
    dest=ROOT/'results'/f'ncu-{label}.log'
    env=dict(os.environ,ROPE_VARIANT=variant,PROFILE_PARENT_LOCK='1')
    cmd=['/usr/local/cuda-13.0/bin/ncu','--kernel-name','regex:sliding','--launch-skip','3','--launch-count','1','--metrics',metrics,'--csv',sys.executable,str(ROOT/'profile.py'),name,'1','1']
    with open('/tmp/okc-gpu.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with dest.open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
    print(dest.name,r.returncode,flush=True)
    if r.returncode:raise SystemExit(r.returncode)

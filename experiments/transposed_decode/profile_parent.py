import fcntl,os,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
metrics=['smsp__inst_executed.sum','l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum','l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum','dram__bytes_read.sum','smsp__warp_issue_stalled_barrier_per_warp_active.pct','smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct','smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct']
with open('/tmp/okc-gpu.lock','w') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX)
 for mode in ('transpose','natural'):
  env={**os.environ,'S35_PROFILE_LOCKED':'1','TDEC_VARIANT':mode}
  with (root/(mode+'-ncu.log')).open('w') as out:
   subprocess.run(['/usr/local/cuda-13.0/bin/ncu','--target-processes','all','--kernel-name','regex:decode','--launch-count','1','--metrics',','.join(metrics),'--csv',sys.executable,str(root/'profile.py')],stdout=out,stderr=subprocess.STDOUT,env=env,check=True)

from pathlib import Path
import subprocess,os,sys
p=Path('experiments/producer_convert')
metrics='gpu__time_duration.sum,launch__registers_per_thread,launch__shared_mem_per_block,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.avg.per_cycle_active,dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_barrier_per_warp_active.pct'
for variant in ('baseline','producer','async_convert'):
 with (p/f'profile-{variant}.log').open('w') as f:
  subprocess.run(['/usr/local/cuda-13.0/bin/ncu','--kernel-name','regex:verify','--launch-skip','2','--launch-count','1','--metrics',metrics,sys.executable,str(p/'profile.py')],env={**os.environ,'VARIANT':variant},stdout=f,stderr=subprocess.STDOUT,check=True)
 print(variant,'profiled',flush=True)

import os,subprocess,sys
from pathlib import Path
p=Path(__file__).resolve().parent
for variant in ('baseline','rawv','key64','rawvk64','paired128'):
 cmd=['/usr/local/cuda/bin/ncu','--target-processes','all','--kernel-name','regex:verify','--launch-skip','2','--launch-count','1','--section','LaunchStats','--section','Occupancy','--section','SchedulerStats','--section','WarpStateStats','--metrics','l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',sys.executable,str(p/'profile.py')]
 with (p/'results'/('profile-'+variant+'.log')).open('w') as f:
  r=subprocess.run(cmd,env={**os.environ,'VARIANT':variant},stdout=f,stderr=subprocess.STDOUT)
 print(variant,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)

for variant in ('baseline','rawv','key64','rawvk64','paired128'):
 cmd=['/usr/local/cuda/bin/ncu','--target-processes','all','--kernel-name','regex:verify','--launch-skip','2','--launch-count','1','--section','SourceCounters','--warp-sampling-interval','0','--import-source','no','--page','source','--csv','--print-source','sass',sys.executable,str(p/'profile.py')]
 target=p/'results'/('source-'+variant+'.csv')
 with target.open('w') as f:
  r=subprocess.run(cmd,env={**os.environ,'VARIANT':variant},stdout=f,stderr=subprocess.STDOUT)
 print('source',variant,r.returncode,flush=True)
 if r.returncode:raise SystemExit(r.returncode)
 with (p/'results'/('source-'+variant+'-summary.json')).open('w') as f:
  subprocess.run([sys.executable,str(p/'source_summary.py'),str(target)],stdout=f,check=True)

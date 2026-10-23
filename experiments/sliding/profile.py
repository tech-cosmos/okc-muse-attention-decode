import fcntl,subprocess
with open('/tmp/okc-gpu.lock','w') as f:
 fcntl.flock(f,fcntl.LOCK_EX)
 subprocess.run(['/usr/local/cuda-13.0/bin/ncu','--set','full','--kernel-name','regex:sliding','--launch-skip','2','--launch-count','1','-f','-o','experiments/sliding/results/warp-v1','/home/daytona/okc/bin/python','experiments/sliding/profile_one.py'],check=True)

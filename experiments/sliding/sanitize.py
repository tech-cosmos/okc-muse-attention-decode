import fcntl,subprocess,sys
with open('/tmp/okc-gpu.lock','w') as f:
 fcntl.flock(f,fcntl.LOCK_EX)
 subprocess.run(['/usr/local/cuda-13.0/bin/compute-sanitizer','--tool',sys.argv[1],'--error-exitcode','42','/home/daytona/okc/bin/python','experiments/sliding/debug_verify.py'],check=True)

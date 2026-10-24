"""Sequential GPU work; no overlap between timing, correctness, and sanitizers."""
import os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def run(name,cmd,env):
    with (ROOT/'results'/name).open('w') as f:
        result=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
    print(name,result.returncode,flush=True)
    assert result.returncode==0,name
base=dict(os.environ)
for variant in ('candidate','compensated'):
    env=dict(base,ROPE_VARIANT=variant)
    if variant=='compensated':
        for mode,args in [('suite',[]),('compare',['--rounds','60','--name','compensated-initial']),('fresh',['--name','compensated-fresh'])]:
            run('compensated-'+mode+'.log',[sys.executable,str(ROOT/'run.py'),mode,*args],env)
    run(variant+'-config.log',[sys.executable,str(ROOT/'config_check.py')],env)
    for tool in ('memcheck','synccheck','racecheck'):
        run(variant+'-'+tool+'.log',['/usr/local/cuda-13.0/bin/compute-sanitizer','--tool',tool,'--error-exitcode','99',sys.executable,str(ROOT/'sanitize.py')],env)

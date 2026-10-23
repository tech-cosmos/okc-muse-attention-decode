"""Validation runner for experimental native replacements, never judge imports."""
import sys,runpy,fcntl
sys.path[:0]=['.','experiments/reduce','experiments/reduce/session5']
import submission,pregate,cooperative
which,mode=sys.argv[1:3]
submission._NATIVE_MERGE=cooperative.MOD if which=="cooperative" else pregate.mods[which]
if mode in ('full','fresh'):
 sys.argv=['validate.py',mode,'production','8']
 runpy.run_path('experiments/reduce/validate.py',run_name='__main__')
elif mode=='partials':
 runpy.run_path('experiments/reduce/stress_final.py',run_name='__main__')
else:raise ValueError(mode)

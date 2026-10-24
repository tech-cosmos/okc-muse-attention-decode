import sys,importlib
sys.path.insert(0,'.')
for name in ('warp_swizzle','cooperative_swizzle','hybrid_swizzle'):
 importlib.import_module('experiments.sliding_warp.'+name)
